"""
推理相关API
"""
import logging
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form
from pydantic import BaseModel

from config import UPLOAD_DIR, OUTPUT_DIR, ALLOWED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)
router = APIRouter()


class InferenceRequest(BaseModel):
    """推理请求"""
    image_path: str
    plugin_name: Optional[str] = None
    ode_steps: Optional[int] = 50
    device: Optional[str] = None
    custom_params: Optional[dict] = {}


class BatchInferenceRequest(BaseModel):
    """批量推理请求"""
    image_paths: List[str]
    plugin_name: Optional[str] = None
    ode_steps: Optional[int] = 50
    device: Optional[str] = None
    custom_params: Optional[dict] = {}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片
    
    Returns:
        {
            'success': bool,
            'file_path': str,  # 相对路径
            'file_name': str,
            'file_size': int,
            'message': str
        }
    """
    try:
        # 检查文件扩展名
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图片格式: {file_ext}，支持的格式: {ALLOWED_IMAGE_EXTENSIONS}"
            )
        
        # 保存文件
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size
        
        logger.info(f"上传图片成功: {file.filename} ({file_size} bytes)")
        
        return {
            'success': True,
            'file_path': str(file_path.relative_to(UPLOAD_DIR.parent)),
            'file_name': file.filename,
            'file_size': file_size,
            'message': '上传成功'
        }
        
    except Exception as e:
        logger.error(f"上传图片失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/upload_batch")
async def upload_batch_images(files: List[UploadFile] = File(...)):
    """
    批量上传图片
    
    Returns:
        {
            'success': bool,
            'total': int,
            'succeeded': int,
            'failed': int,
            'files': List[dict],
            'message': str
        }
    """
    results = []
    succeeded = 0
    failed = 0
    
    for file in files:
        try:
            # 检查文件扩展名
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                results.append({
                    'file_name': file.filename,
                    'success': False,
                    'message': f'不支持的格式: {file_ext}'
                })
                failed += 1
                continue
            
            # 保存文件
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = file_path.stat().st_size
            
            results.append({
                'file_name': file.filename,
                'file_path': str(file_path.relative_to(UPLOAD_DIR.parent)),
                'file_size': file_size,
                'success': True,
                'message': '上传成功'
            })
            succeeded += 1
            
        except Exception as e:
            logger.error(f"上传 {file.filename} 失败: {e}")
            results.append({
                'file_name': file.filename,
                'success': False,
                'message': str(e)
            })
            failed += 1
    
    logger.info(f"批量上传完成: 总计={len(files)}, 成功={succeeded}, 失败={failed}")
    
    return {
        'success': True,
        'total': len(files),
        'succeeded': succeeded,
        'failed': failed,
        'files': results,
        'message': f'上传完成: {succeeded}/{len(files)} 成功'
    }


@router.post("/infer")
async def infer_single_image(request: Request, req_body: InferenceRequest):
    """
    单张图片推理
    
    Returns:
        {
            'success': bool,
            'output_path': str,
            'inference_time': float,
            'message': str
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        # 检查输入文件是否存在
        input_path = Path(req_body.image_path)
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"输入文件不存在: {req_body.image_path}")
        
        # 确定输出路径
        output_filename = f"output_{input_path.stem}.png"
        output_path = OUTPUT_DIR / output_filename
        
        # 执行推理
        result = model_manager.inference(
            image_path=str(input_path),
            output_path=str(output_path),
            plugin_name=req_body.plugin_name,
            ode_steps=req_body.ode_steps,
            **req_body.custom_params
        )
        
        # 如果成功，转换输出路径为相对路径（用于前端访问）
        if result['success'] and 'output_path' in result:
            result['output_url'] = f"/outputs/{output_filename}"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@router.post("/infer_batch")
async def infer_batch_images(request: Request, req_body: BatchInferenceRequest):
    """
    批量推理
    
    Returns:
        {
            'success': bool,
            'total': int,
            'succeeded': int,
            'failed': int,
            'results': List[dict],
            'total_time': float,
            'message': str
        }
    """
    model_manager = request.app.state.model_manager
    
    try:
        # 创建输出目录
        batch_output_dir = OUTPUT_DIR / "batch"
        batch_output_dir.mkdir(exist_ok=True)
        
        # 执行批量推理
        result = model_manager.batch_inference(
            image_paths=req_body.image_paths,
            output_dir=str(batch_output_dir),
            plugin_name=req_body.plugin_name,
            ode_steps=req_body.ode_steps,
            **req_body.custom_params
        )
        
        # 为每个结果添加输出URL
        if result['success'] and 'results' in result:
            for item in result['results']:
                if item.get('success') and 'output_path' in item:
                    output_path = Path(item['output_path'])
                    item['output_url'] = f"/outputs/batch/{output_path.name}"
        
        return result
        
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")


@router.get("/list_uploaded")
async def list_uploaded_images():
    """
    列出所有已上传的图片
    
    Returns:
        {
            'success': bool,
            'images': List[dict],
            'total': int
        }
    """
    try:
        images = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                images.append({
                    'file_name': file_path.name,
                    'file_path': str(file_path.relative_to(UPLOAD_DIR.parent)),
                    'file_size': file_path.stat().st_size,
                    'modified_time': file_path.stat().st_mtime
                })
        
        # 按修改时间降序排序
        images.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return {
            'success': True,
            'images': images,
            'total': len(images)
        }
        
    except Exception as e:
        logger.error(f"列出上传图片失败: {e}")
        raise HTTPException(status_code=500, detail=f"操作失败: {str(e)}")


@router.get("/list_outputs")
async def list_output_images():
    """
    列出所有输出图片
    
    Returns:
        {
            'success': bool,
            'images': List[dict],
            'total': int
        }
    """
    try:
        images = []
        
        # 单张推理输出
        for file_path in OUTPUT_DIR.glob("*.png"):
            if file_path.is_file():
                images.append({
                    'file_name': file_path.name,
                    'file_path': str(file_path.relative_to(OUTPUT_DIR.parent)),
                    'output_url': f"/outputs/{file_path.name}",
                    'file_size': file_path.stat().st_size,
                    'created_time': file_path.stat().st_ctime
                })
        
        # 批量推理输出
        batch_dir = OUTPUT_DIR / "batch"
        if batch_dir.exists():
            for file_path in batch_dir.glob("*.png"):
                if file_path.is_file():
                    images.append({
                        'file_name': file_path.name,
                        'file_path': str(file_path.relative_to(OUTPUT_DIR.parent)),
                        'output_url': f"/outputs/batch/{file_path.name}",
                        'file_size': file_path.stat().st_size,
                        'created_time': file_path.stat().st_ctime
                    })
        
        # 按创建时间降序排序
        images.sort(key=lambda x: x['created_time'], reverse=True)
        
        return {
            'success': True,
            'images': images,
            'total': len(images)
        }
        
    except Exception as e:
        logger.error(f"列出输出图片失败: {e}")
        raise HTTPException(status_code=500, detail=f"操作失败: {str(e)}")


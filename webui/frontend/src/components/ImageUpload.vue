<template>
  <div class="image-upload">
    <!-- 上传区域 -->
    <el-upload
      class="upload-area"
      drag
      :auto-upload="false"
      :on-change="handleFileChange"
      :show-file-list="false"
      accept="image/png,image/jpeg,image/jpg,image/bmp"
    >
      <el-icon class="upload-icon"><Upload /></el-icon>
      <div class="upload-text">拖拽图片到此处 或 <em>点击上传</em></div>
      <div class="upload-tip">支持: PNG, JPG, JPEG, BMP</div>
    </el-upload>

    <!-- 预览区域 -->
    <div v-if="inferenceStore.currentInputImage" class="preview-area">
      <div class="preview-label">当前图片:</div>
      <img 
        :src="inferenceStore.currentInputImage.file_url" 
        class="preview-image"
        alt="输入图片"
      />
      <div class="preview-info">
        {{ inferenceStore.currentInputImage.file_name }}
      </div>
    </div>

    <!-- 推理按钮 -->
    <el-button 
      type="primary" 
      size="large"
      class="infer-button"
      :loading="inferenceStore.inferring"
      :disabled="!inferenceStore.currentInputImage || !modelStore.isModelLoaded"
      @click="handleInference"
    >
      <el-icon v-if="!inferenceStore.inferring"><Lightning /></el-icon>
      {{ inferenceStore.inferring ? '推理中...' : '开始推理' }}
    </el-button>

    <!-- 提示信息 -->
    <el-alert
      v-if="!modelStore.isModelLoaded"
      type="warning"
      :closable="false"
      show-icon
      style="margin-top: 15px;"
    >
      <template #title>
        请先加载模型
      </template>
    </el-alert>
  </div>
</template>

<script setup>
import { useInferenceStore } from '@/stores/inference'
import { useModelStore } from '@/stores/model'

const inferenceStore = useInferenceStore()
const modelStore = useModelStore()

const handleFileChange = (file) => {
  inferenceStore.uploadSingleImage(file.raw)
}

const handleInference = () => {
  inferenceStore.runInference()
}
</script>

<style scoped>
.image-upload {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.upload-area {
  width: 100%;
}

:deep(.el-upload) {
  width: 100%;
}

:deep(.el-upload-dragger) {
  width: 100%;
  padding: 30px 20px;
}

.upload-icon {
  font-size: 67px;
  color: #409eff;
  margin-bottom: 16px;
}

.upload-text {
  font-size: 16px;
  color: #606266;
  margin-bottom: 8px;
}

.upload-text em {
  color: #409eff;
  font-style: normal;
}

.upload-tip {
  font-size: 12px;
  color: #909399;
}

.preview-area {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 6px;
}

.preview-label {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.preview-image {
  width: 100%;
  height: auto;
  border-radius: 4px;
  border: 1px solid #dcdfe6;
}

.preview-info {
  font-size: 12px;
  color: #909399;
  text-align: center;
  word-break: break-all;
}

.infer-button {
  width: 100%;
  height: 45px;
  font-size: 16px;
  font-weight: 600;
}
</style>


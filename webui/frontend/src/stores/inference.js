/**
 * 推理状态管理
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { 
  uploadImage,
  uploadBatchImages,
  inferSingleImage,
  inferBatchImages
} from '@/api/inference'
import { ElMessage } from 'element-plus'

export const useInferenceStore = defineStore('inference', () => {
  // 状态
  const uploadedImages = ref([])
  const currentInputImage = ref(null)
  const currentOutputImage = ref(null)
  const inferring = ref(false)
  const inferenceTime = ref(0)
  const inferenceParams = ref({
    ode_steps: 50,
    device: 'cuda:0',
    custom_params: {}
  })

  // 批量推理状态
  const batchResults = ref([])
  const batchProgress = ref(0)

  // 方法
  async function uploadSingleImage(file) {
    try {
      const response = await uploadImage(file)
      
      if (response.success) {
        currentInputImage.value = {
          file_name: response.file_name,
          file_path: response.file_path,
          file_url: URL.createObjectURL(file)
        }
        uploadedImages.value.unshift(currentInputImage.value)
        ElMessage.success('上传成功')
        return true
      } else {
        ElMessage.error(response.message || '上传失败')
        return false
      }
    } catch (error) {
      ElMessage.error('上传失败')
      console.error(error)
      return false
    }
  }

  async function uploadMultipleImages(files) {
    try {
      const response = await uploadBatchImages(files)
      
      if (response.success) {
        ElMessage.success(`上传完成: ${response.succeeded}/${response.total}`)
        return response
      } else {
        ElMessage.error('批量上传失败')
        return null
      }
    } catch (error) {
      ElMessage.error('批量上传失败')
      console.error(error)
      return null
    }
  }

  async function runInference() {
    if (!currentInputImage.value) {
      ElMessage.warning('请先上传图片')
      return false
    }

    inferring.value = true
    currentOutputImage.value = null
    
    try {
      const response = await inferSingleImage({
        image_path: currentInputImage.value.file_path,
        plugin_name: null, // 使用当前插件
        ode_steps: inferenceParams.value.ode_steps,
        device: inferenceParams.value.device,
        custom_params: inferenceParams.value.custom_params
      })

      if (response.success) {
        currentOutputImage.value = {
          output_url: response.output_url,
          output_path: response.output_path
        }
        inferenceTime.value = response.inference_time
        ElMessage.success(`推理成功 (${response.inference_time.toFixed(2)}s)`)
        return true
      } else {
        ElMessage.error(response.message || '推理失败')
        return false
      }
    } catch (error) {
      ElMessage.error('推理失败')
      console.error(error)
      return false
    } finally {
      inferring.value = false
    }
  }

  async function runBatchInference(imagePaths) {
    inferring.value = true
    batchResults.value = []
    batchProgress.value = 0

    try {
      const response = await inferBatchImages({
        image_paths: imagePaths,
        plugin_name: null,
        ode_steps: inferenceParams.value.ode_steps,
        device: inferenceParams.value.device,
        custom_params: inferenceParams.value.custom_params
      })

      if (response.success) {
        batchResults.value = response.results
        batchProgress.value = 100
        ElMessage.success(
          `批量推理完成: ${response.succeeded}/${response.total} (${response.total_time.toFixed(2)}s)`
        )
        return response
      } else {
        ElMessage.error(response.message || '批量推理失败')
        return null
      }
    } catch (error) {
      ElMessage.error('批量推理失败')
      console.error(error)
      return null
    } finally {
      inferring.value = false
    }
  }

  function setCurrentInput(image) {
    currentInputImage.value = image
    currentOutputImage.value = null
  }

  function updateInferenceParams(params) {
    inferenceParams.value = { ...inferenceParams.value, ...params }
  }

  function clearResults() {
    currentInputImage.value = null
    currentOutputImage.value = null
    batchResults.value = []
    inferenceTime.value = 0
  }

  return {
    // 状态
    uploadedImages,
    currentInputImage,
    currentOutputImage,
    inferring,
    inferenceTime,
    inferenceParams,
    batchResults,
    batchProgress,
    // 方法
    uploadSingleImage,
    uploadMultipleImages,
    runInference,
    runBatchInference,
    setCurrentInput,
    updateInferenceParams,
    clearResults
  }
})


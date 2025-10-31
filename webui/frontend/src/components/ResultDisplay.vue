<template>
  <div class="result-display">
    <!-- 对比视图 -->
    <div v-if="hasResults" class="comparison-view">
      <!-- 输入图 -->
      <div class="image-panel">
        <div class="panel-header">
          <span class="panel-title">输入图 (Sim)</span>
        </div>
        <div class="image-container">
          <img 
            :src="inferenceStore.currentInputImage.file_url" 
            alt="输入图"
            class="result-image"
          />
        </div>
      </div>

      <!-- 输出图 -->
      <div class="image-panel">
        <div class="panel-header">
          <span class="panel-title">输出图 (Real)</span>
          <el-tag v-if="inferenceStore.inferenceTime > 0" type="success" size="small">
            {{ inferenceStore.inferenceTime.toFixed(2) }}s
          </el-tag>
        </div>
        <div class="image-container">
          <img 
            v-if="inferenceStore.currentOutputImage"
            :src="inferenceStore.currentOutputImage.output_url" 
            alt="输出图"
            class="result-image"
          />
          <div v-else class="loading-placeholder">
            <el-icon class="is-loading"><Loading /></el-icon>
            <div>推理中...</div>
          </div>
        </div>
        <div v-if="inferenceStore.currentOutputImage" class="image-actions">
          <el-button 
            type="primary" 
            size="small"
            @click="downloadImage"
          >
            <el-icon><Download /></el-icon>
            下载
          </el-button>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-else class="empty-state">
      <el-icon class="empty-icon"><Picture /></el-icon>
      <div class="empty-text">暂无推理结果</div>
      <div class="empty-tip">请上传图片并点击"开始推理"</div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useInferenceStore } from '@/stores/inference'
import { ElMessage } from 'element-plus'

const inferenceStore = useInferenceStore()

const hasResults = computed(() => {
  return inferenceStore.currentInputImage !== null
})

const downloadImage = () => {
  if (!inferenceStore.currentOutputImage) return
  
  const url = inferenceStore.currentOutputImage.output_url
  const link = document.createElement('a')
  link.href = url
  link.download = `output_${Date.now()}.png`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  ElMessage.success('下载成功')
}
</script>

<style scoped>
.result-display {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.comparison-view {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  height: 100%;
}

.image-panel {
  display: flex;
  flex-direction: column;
  background: #f5f7fa;
  border-radius: 8px;
  padding: 15px;
  overflow: hidden;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 2px solid #e4e7ed;
}

.panel-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.image-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #fff;
  border-radius: 6px;
  overflow: hidden;
  min-height: 400px;
}

.result-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.loading-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
  color: #909399;
  font-size: 14px;
}

.loading-placeholder .el-icon {
  font-size: 40px;
}

.image-actions {
  display: flex;
  justify-content: center;
  margin-top: 12px;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #909399;
}

.empty-icon {
  font-size: 100px;
  margin-bottom: 20px;
  color: #c0c4cc;
}

.empty-text {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 10px;
}

.empty-tip {
  font-size: 14px;
}

/* 响应式 */
@media (max-width: 1200px) {
  .comparison-view {
    grid-template-columns: 1fr;
  }
}
</style>


<template>
  <div class="model-selector-container">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <span class="title">
            <el-icon><Box /></el-icon>
            模型选择
          </span>
          <el-button 
            type="primary" 
            :icon="Refresh" 
            size="small"
            @click="handleScan"
            :loading="scanning"
          >
            {{ scanning ? '扫描中...' : '扫描模型' }}
          </el-button>
        </div>
      </template>

      <!-- 模型列表为空时的提示 -->
      <el-empty 
        v-if="!loading && models.length === 0" 
        description="未找到模型，请先训练模型并放入trained_models目录"
      >
        <el-button type="primary" @click="handleScan">立即扫描</el-button>
      </el-empty>

      <!-- 加载中 -->
      <div v-if="loading" class="loading-container">
        <el-icon class="is-loading"><Loading /></el-icon>
        <span>正在加载模型列表...</span>
      </div>

      <!-- 模型选择器 -->
      <div v-if="!loading && models.length > 0" class="selector-content">
        <!-- 下拉选择 -->
        <div class="select-group">
          <label>选择模型：</label>
          <el-select
            v-model="selectedModelId"
            placeholder="请选择一个模型"
            filterable
            size="large"
            style="width: 100%"
            @change="handleModelChange"
          >
            <el-option-group
              v-for="group in modelGroups"
              :key="group.project"
              :label="group.project"
            >
              <el-option
                v-for="model in group.models"
                :key="model.id"
                :label="model.name"
                :value="model.id"
              >
                <div class="model-option">
                  <span class="model-name">{{ model.name }}</span>
                  <div class="model-meta">
                    <el-tag v-if="model.epoch !== 'unknown'" size="small" type="info">
                      Epoch: {{ model.epoch }}
                    </el-tag>
                    <el-tag v-if="model.val_loss !== 'unknown'" size="small" type="success">
                      Loss: {{ typeof model.val_loss === 'number' ? model.val_loss.toFixed(4) : model.val_loss }}
                    </el-tag>
                    <el-tag v-if="!model.has_config" size="small" type="warning">
                      无配置
                    </el-tag>
                    <el-tag v-if="model.has_config" size="small" type="success">
                      ✓ 有配置
                    </el-tag>
                  </div>
                </div>
              </el-option>
            </el-option-group>
          </el-select>
        </div>

        <!-- 选中模型的详细信息 -->
        <div v-if="selectedModel" class="model-details">
          <el-descriptions :column="2" border>
            <el-descriptions-item label="模型ID">
              {{ selectedModel.id }}
            </el-descriptions-item>
            <el-descriptions-item label="项目">
              {{ selectedModel.project }}
            </el-descriptions-item>
            <el-descriptions-item label="文件名">
              {{ selectedModel.name }}
            </el-descriptions-item>
            <el-descriptions-item label="Epoch">
              {{ selectedModel.epoch }}
            </el-descriptions-item>
            <el-descriptions-item label="验证Loss">
              {{ typeof selectedModel.val_loss === 'number' ? selectedModel.val_loss.toFixed(6) : selectedModel.val_loss }}
            </el-descriptions-item>
            <el-descriptions-item label="配置文件">
              <el-tag v-if="selectedModel.has_config" type="success">✓ 有</el-tag>
              <el-tag v-else type="warning">✗ 无</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="路径" :span="2">
              <el-text size="small" type="info">{{ selectedModel.path }}</el-text>
            </el-descriptions-item>
          </el-descriptions>

          <!-- 加载/卸载按钮 -->
          <div class="action-buttons">
            <el-button
              v-if="!isCurrentModelLoaded"
              type="primary"
              size="large"
              @click="handleLoadModel"
              :loading="loadModelLoading"
              :disabled="!selectedModelId"
            >
              {{ loadModelLoading ? '加载中...' : '加载模型' }}
            </el-button>
            <el-button
              v-if="isCurrentModelLoaded"
              type="warning"
              size="large"
              @click="handleUnloadCurrentModel"
              :loading="unloadModelLoading"
            >
              {{ unloadModelLoading ? '卸载中...' : '卸载当前模型' }}
            </el-button>
          </div>
        </div>

        <!-- 统计信息 -->
        <div class="stats">
          <el-text type="info" size="small">
            共找到 {{ models.length }} 个模型，来自 {{ Object.keys(modelGroups).length }} 个项目
          </el-text>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElNotification, ElMessageBox } from 'element-plus'
import { Box, Refresh, Loading } from '@element-plus/icons-vue'
import { useModelStore } from '@/stores/model'
import axios from 'axios'

// 使用相对路径，会通过Vite代理转发到后端
const API_BASE_URL = ''

// Store
const modelStore = useModelStore()

// 状态
const models = ref([])
const selectedModelId = ref(null)
const loading = ref(false)
const scanning = ref(false)
const loadModelLoading = ref(false)
const unloadModelLoading = ref(false)

// 计算属性
const selectedModel = computed(() => {
  if (!selectedModelId.value) return null
  return models.value.find(m => m.id === selectedModelId.value)
})

// 按项目分组模型
const modelGroups = computed(() => {
  const groups = {}
  models.value.forEach(model => {
    if (!groups[model.project]) {
      groups[model.project] = {
        project: model.project,
        models: []
      }
    }
    groups[model.project].models.push(model)
  })
  return groups
})

// 检查当前是否有模型已加载
const isCurrentModelLoaded = computed(() => {
  return modelStore.isModelLoaded
})

// 获取已扫描的模型列表
async function fetchModels() {
  loading.value = true
  try {
    const response = await axios.get(`${API_BASE_URL}/api/models/scanned`)
    if (response.data.success) {
      models.value = response.data.models
      console.log(`获取到 ${models.value.length} 个模型`)
    }
  } catch (error) {
    console.error('获取模型列表失败:', error)
    ElMessage.error('获取模型列表失败')
  } finally {
    loading.value = false
  }
}

// 扫描trained_models目录
async function handleScan() {
  scanning.value = true
  try {
    const response = await axios.get(`${API_BASE_URL}/api/models/scan`)
    if (response.data.success) {
      models.value = response.data.models
      ElNotification({
        title: '扫描完成',
        message: `发现 ${response.data.total} 个模型`,
        type: 'success'
      })
    }
  } catch (error) {
    console.error('扫描失败:', error)
    ElMessage.error('扫描失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    scanning.value = false
  }
}

// 选择模型变化
function handleModelChange(modelId) {
  console.log('选中模型:', modelId)
}

// 加载模型
async function handleLoadModel() {
  if (!selectedModelId.value) {
    ElMessage.warning('请先选择一个模型')
    return
  }

  loadModelLoading.value = true
  try {
    const response = await axios.post(`${API_BASE_URL}/api/models/auto_load`, {
      model_id: selectedModelId.value,
      device: 'cuda:0'
    })

    if (response.data.success) {
      // 立即更新modelStore状态
      await modelStore.loadPluginList()
      
      ElNotification({
        title: '加载成功',
        message: `模型 ${selectedModelId.value} 已成功加载`,
        type: 'success',
        duration: 3000
      })

      // 触发事件，通知父组件模型已加载
      emit('model-loaded', {
        modelId: selectedModelId.value,
        pluginName: response.data.plugin_name
      })
    }
  } catch (error) {
    console.error('加载模型失败:', error)
    ElMessage.error('加载模型失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loadModelLoading.value = false
  }
}

// 卸载当前模型
async function handleUnloadCurrentModel() {
  try {
    await ElMessageBox.confirm(
      '确定要卸载当前模型吗？\n卸载后将释放显存，但插件仍然保持注册状态。',
      '确认卸载模型',
      {
        confirmButtonText: '卸载',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    unloadModelLoading.value = true
    
    const success = await modelStore.unloadCurrentModel()
    if (success) {
      // 立即更新状态
      await modelStore.loadPluginList()
      
      ElNotification({
        title: '卸载成功',
        message: '模型已卸载，显存已释放',
        type: 'success',
        duration: 3000
      })
    }
  } catch {
    // 用户取消
  } finally {
    unloadModelLoading.value = false
  }
}

// 发射事件
const emit = defineEmits(['model-loaded'])

// 组件挂载时加载模型列表
onMounted(() => {
  fetchModels()
})

// 暴露方法供父组件调用
defineExpose({
  fetchModels,
  handleScan
})
</script>

<style scoped>
.model-selector-container {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
  gap: 16px;
}

.loading-container .el-icon {
  font-size: 32px;
  color: #409EFF;
}

.selector-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.select-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.select-group label {
  font-weight: 500;
  color: #606266;
}

.model-option {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 4px 0;
}

.model-name {
  font-weight: 500;
  color: #303133;
}

.model-meta {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.model-details {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.action-buttons {
  display: flex;
  justify-content: center;
  margin-top: 8px;
}

.stats {
  text-align: center;
  padding-top: 12px;
  border-top: 1px solid #EBEEF5;
}

:deep(.el-select-dropdown__item) {
  height: auto !important;
  padding: 8px 20px !important;
}
</style>

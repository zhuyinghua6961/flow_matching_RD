<template>
  <div class="model-selector">
    <el-select
      v-model="currentModel"
      placeholder="选择模型插件"
      size="large"
      @change="handleModelChange"
      :loading="modelStore.loading"
    >
      <el-option
        v-for="plugin in modelStore.plugins"
        :key="plugin.name"
        :label="plugin.name"
        :value="plugin.name"
      >
        <div class="plugin-option">
          <span>{{ plugin.name }}</span>
          <el-tag 
            v-if="plugin.is_loaded" 
            type="success" 
            size="small"
            effect="dark"
          >
            已加载
          </el-tag>
          <el-tag 
            v-else
            type="info" 
            size="small"
          >
            未加载
          </el-tag>
        </div>
      </el-option>
    </el-select>

    <el-button
      v-if="modelStore.isModelLoaded"
      type="danger"
      size="large"
      :loading="modelStore.loading"
      @click="handleUnload"
    >
      <el-icon><Close /></el-icon>
      卸载模型
    </el-button>
    <el-button
      v-else
      type="success"
      size="large"
      :loading="modelStore.loading"
      :disabled="!currentModel"
      @click="showLoadDialog = true"
    >
      <el-icon><Check /></el-icon>
      加载模型
    </el-button>

    <!-- 加载模型对话框 -->
    <el-dialog
      v-model="showLoadDialog"
      title="加载模型"
      width="500px"
    >
      <el-form label-width="100px">
        <el-form-item label="检查点路径">
          <el-input
            v-model="checkpointPath"
            placeholder="留空则使用配置中的路径"
          />
        </el-form-item>
        <el-form-item label="设备">
          <el-select v-model="device" style="width: 100%;">
            <el-option label="cuda:0" value="cuda:0" />
            <el-option label="cuda:1" value="cuda:1" />
            <el-option label="cpu" value="cpu" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showLoadDialog = false">取消</el-button>
        <el-button 
          type="primary" 
          @click="handleLoad"
          :loading="modelStore.loading"
        >
          确定
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useModelStore } from '@/stores/model'
import { ElMessageBox } from 'element-plus'

const modelStore = useModelStore()

const currentModel = ref(modelStore.currentPlugin)
const showLoadDialog = ref(false)
const checkpointPath = ref('')
const device = ref('cuda:0')

watch(() => modelStore.currentPlugin, (newVal) => {
  currentModel.value = newVal
})

const handleModelChange = (pluginName) => {
  if (pluginName !== modelStore.currentPlugin) {
    modelStore.switchToPlugin(pluginName)
  }
}

const handleLoad = async () => {
  const success = await modelStore.loadCurrentModel(
    checkpointPath.value || null,
    device.value
  )
  if (success) {
    showLoadDialog.value = false
    checkpointPath.value = ''
  }
}

const handleUnload = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要卸载当前模型吗？',
      '确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    await modelStore.unloadCurrentModel()
  } catch {
    // 用户取消
  }
}
</script>

<style scoped>
.model-selector {
  display: flex;
  align-items: center;
  gap: 15px;
}

.el-select {
  min-width: 200px;
}

.plugin-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}
</style>


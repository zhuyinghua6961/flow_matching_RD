/**
 * 模型状态管理
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { 
  listPlugins, 
  loadModel, 
  unloadModel, 
  switchPlugin,
  getPluginInfo 
} from '@/api/model'
import { ElMessage } from 'element-plus'

export const useModelStore = defineStore('model', () => {
  // 状态
  const plugins = ref([])
  const currentPlugin = ref(null)
  const loading = ref(false)

  // 计算属性
  const currentPluginInfo = computed(() => {
    if (!currentPlugin.value) return null
    return plugins.value.find(p => p.name === currentPlugin.value)
  })

  const isModelLoaded = computed(() => {
    if (!currentPluginInfo.value) return false
    return currentPluginInfo.value.is_loaded
  })

  // 方法
  async function loadPluginList() {
    loading.value = true
    try {
      const response = await listPlugins()
      if (response.success) {
        plugins.value = response.plugins
        currentPlugin.value = response.current_plugin
      }
    } catch (error) {
      ElMessage.error('加载插件列表失败')
      console.error(error)
    } finally {
      loading.value = false
    }
  }

  async function loadCurrentModel(checkpointPath = null, device = 'cuda:0') {
    loading.value = true
    try {
      const response = await loadModel({
        plugin_name: currentPlugin.value,
        checkpoint_path: checkpointPath,
        device: device
      })
      
      if (response.success) {
        ElMessage.success('模型加载成功')
        await loadPluginList() // 刷新插件列表
        return true
      } else {
        ElMessage.error(response.message || '模型加载失败')
        return false
      }
    } catch (error) {
      ElMessage.error('模型加载失败')
      console.error(error)
      return false
    } finally {
      loading.value = false
    }
  }

  async function unloadCurrentModel() {
    loading.value = true
    try {
      const response = await unloadModel(currentPlugin.value)
      
      if (response.success) {
        ElMessage.success('模型卸载成功')
        await loadPluginList()
        return true
      } else {
        ElMessage.error(response.message || '模型卸载失败')
        return false
      }
    } catch (error) {
      ElMessage.error('模型卸载失败')
      console.error(error)
      return false
    } finally {
      loading.value = false
    }
  }

  async function switchToPlugin(pluginName) {
    loading.value = true
    try {
      const response = await switchPlugin(pluginName)
      
      if (response.success) {
        currentPlugin.value = pluginName
        ElMessage.success(`已切换到: ${pluginName}`)
        await loadPluginList()
        return true
      } else {
        ElMessage.error(response.message || '切换插件失败')
        return false
      }
    } catch (error) {
      ElMessage.error('切换插件失败')
      console.error(error)
      return false
    } finally {
      loading.value = false
    }
  }

  async function getModelInfo(pluginName) {
    try {
      const response = await getPluginInfo(pluginName)
      return response
    } catch (error) {
      console.error(error)
      return null
    }
  }

  return {
    // 状态
    plugins,
    currentPlugin,
    loading,
    // 计算属性
    currentPluginInfo,
    isModelLoaded,
    // 方法
    loadPluginList,
    loadCurrentModel,
    unloadCurrentModel,
    switchToPlugin,
    getModelInfo
  }
})


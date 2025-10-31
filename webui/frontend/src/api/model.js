/**
 * 模型管理相关API
 */
import api from './inference'

/**
 * 列出所有插件
 * @returns {Promise}
 */
export function listPlugins() {
  return api.get('/models/list')
}

/**
 * 上传插件文件
 * @param {File} file - 插件文件
 * @returns {Promise}
 */
export function uploadPlugin(file) {
  const formData = new FormData()
  formData.append('file', file)
  
  return api.post('/models/upload_plugin', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

/**
 * 注册插件
 * @param {Object} data - 插件信息
 * @returns {Promise}
 */
export function registerPlugin(data) {
  return api.post('/models/register', data)
}

/**
 * 加载模型
 * @param {Object} data - 加载参数
 * @returns {Promise}
 */
export function loadModel(data) {
  return api.post('/models/load', data)
}

/**
 * 卸载模型
 * @param {string} pluginName - 插件名称
 * @returns {Promise}
 */
export function unloadModel(pluginName = null) {
  return api.post('/models/unload', null, {
    params: { plugin_name: pluginName }
  })
}

/**
 * 切换插件
 * @param {string} pluginName - 插件名称
 * @returns {Promise}
 */
export function switchPlugin(pluginName) {
  return api.post('/models/switch', { plugin_name: pluginName })
}

/**
 * 获取插件信息
 * @param {string} pluginName - 插件名称
 * @returns {Promise}
 */
export function getPluginInfo(pluginName) {
  return api.get(`/models/info/${pluginName}`)
}

/**
 * 注销插件
 * @param {string} pluginName - 插件名称
 * @returns {Promise}
 */
export function unregisterPlugin(pluginName) {
  return api.delete(`/models/unregister/${pluginName}`)
}


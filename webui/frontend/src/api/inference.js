/**
 * 推理相关API
 */
import axios from 'axios'

// 创建axios实例
const api = axios.create({
  baseURL: '/api',
  timeout: 300000 // 5分钟超时（推理可能较慢）
})

// 请求拦截器
api.interceptors.request.use(
  config => {
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    console.error('API请求失败:', error)
    return Promise.reject(error)
  }
)

/**
 * 上传单张图片
 * @param {File} file - 图片文件
 * @returns {Promise}
 */
export function uploadImage(file) {
  const formData = new FormData()
  formData.append('file', file)
  
  return api.post('/inference/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

/**
 * 批量上传图片
 * @param {File[]} files - 图片文件数组
 * @returns {Promise}
 */
export function uploadBatchImages(files) {
  const formData = new FormData()
  files.forEach(file => {
    formData.append('files', file)
  })
  
  return api.post('/inference/upload_batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

/**
 * 单张图片推理
 * @param {Object} data - 推理参数
 * @returns {Promise}
 */
export function inferSingleImage(data) {
  return api.post('/inference/infer', data)
}

/**
 * 批量推理
 * @param {Object} data - 批量推理参数
 * @returns {Promise}
 */
export function inferBatchImages(data) {
  return api.post('/inference/infer_batch', data)
}

/**
 * 列出已上传的图片
 * @returns {Promise}
 */
export function listUploadedImages() {
  return api.get('/inference/list_uploaded')
}

/**
 * 列出输出图片
 * @returns {Promise}
 */
export function listOutputImages() {
  return api.get('/inference/list_outputs')
}

export default api


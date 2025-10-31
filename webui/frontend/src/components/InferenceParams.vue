<template>
  <div class="inference-params">
    <el-form label-width="100px" label-position="left">
      <el-form-item label="ODE步数">
        <el-slider
          v-model="params.ode_steps"
          :min="10"
          :max="100"
          :step="5"
          show-input
          :show-input-controls="false"
        />
        <div class="param-tip">步数越多，质量越好，但速度越慢</div>
      </el-form-item>

      <el-form-item label="设备">
        <el-select v-model="params.device" style="width: 100%;">
          <el-option label="CUDA:0 (GPU 0)" value="cuda:0" />
          <el-option label="CUDA:1 (GPU 1)" value="cuda:1" />
          <el-option label="CPU" value="cpu" />
        </el-select>
      </el-form-item>

      <el-divider />

      <!-- 自定义参数（可扩展） -->
      <el-collapse v-model="activeCollapse">
        <el-collapse-item title="高级参数" name="advanced">
          <el-form-item label="ODE方法">
            <el-select 
              v-model="customParams.ode_method" 
              style="width: 100%;"
              placeholder="默认: euler"
            >
              <el-option label="Euler" value="euler" />
              <el-option label="RK4 (未实现)" value="rk4" disabled />
            </el-select>
          </el-form-item>

          <el-alert
            type="info"
            :closable="false"
            show-icon
          >
            <template #title>
              更多参数请参考插件文档
            </template>
          </el-alert>
        </el-collapse-item>
      </el-collapse>
    </el-form>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useInferenceStore } from '@/stores/inference'

const inferenceStore = useInferenceStore()

const params = ref({
  ode_steps: inferenceStore.inferenceParams.ode_steps,
  device: inferenceStore.inferenceParams.device
})

const customParams = ref({
  ode_method: 'euler'
})

const activeCollapse = ref([])

// 监听参数变化，更新到store
watch(params, (newParams) => {
  inferenceStore.updateInferenceParams({
    ...newParams,
    custom_params: customParams.value
  })
}, { deep: true })

watch(customParams, (newCustomParams) => {
  inferenceStore.updateInferenceParams({
    custom_params: newCustomParams
  })
}, { deep: true })
</script>

<style scoped>
.inference-params {
  width: 100%;
}

.param-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 8px;
  line-height: 1.4;
}

:deep(.el-slider__input) {
  width: 80px;
}

:deep(.el-form-item) {
  margin-bottom: 22px;
}

:deep(.el-collapse-item__header) {
  font-weight: 600;
}
</style>


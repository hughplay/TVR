<template>
  <a-layout class="layout">
    <a-layout-header>
      <a-menu
        v-model:selectedKeys="selectedKeys"
        theme="dark"
        mode="horizontal"
        :style="{ lineHeight: '64px' }"
        class="flex"
      >
        <a-menu-item @click="$router.push('/')" key="viewer"
          >TRANCE</a-menu-item
        >
        <!-- <a-menu-item @click="$router.push('/analyze')" key="analyze"
          >Analysis</a-menu-item
        >
        <a-menu-item @click="$router.push('/compare')" key="compare"
          >Comparison</a-menu-item
        > -->
        <!-- <a-menu-item key="2">CASP15</a-menu-item>
        <a-menu-item key="3">nav 3</a-menu-item> -->
      </a-menu>
    </a-layout-header>
    <a-layout-content style="padding: 0 50px">
      <div class="flex justify-between items-center py-4">
        <a-breadcrumb>
          <a-breadcrumb-item>Home</a-breadcrumb-item>
          <a-breadcrumb-item v-if="$route.path == '/'"
            >TRANCE</a-breadcrumb-item
          >
          <!-- <a-breadcrumb-item v-if="$route.path == '/analyze'"
            >Analysis</a-breadcrumb-item
          >
          <a-breadcrumb-item v-if="$route.path == '/compare'"
            >Comparison</a-breadcrumb-item
          > -->
        </a-breadcrumb>
      </div>
      <div :style="{ background: '#fff', padding: '24px', minHeight: '280px' }">
        <router-view />
      </div>
    </a-layout-content>
    <a-layout-footer style="text-align: center">
      TVR ©2023
    </a-layout-footer>
  </a-layout>
</template>

<script>
import { LinkOutlined, ToolOutlined } from "@ant-design/icons-vue";
export default {
  mounted: function () {},
  data() {
    return {
      selectedKeys: ["viewer"],
    };
  },
  watch: {
    $route: function (new_route) {
      if (new_route.path == "/") {
        this.selectedKeys = ["viewer"];
      } else if (new_route.path == "/analyze") {
        this.selectedKeys = ["analyze"];
      } else if (new_route.path == "/compare") {
        this.selectedKeys = ["compare"];
      }
    },
  },
  components: {
    LinkOutlined,
    ToolOutlined,
  },
};
</script>
<style>
.site-layout-content {
  min-height: 280px;
  padding: 24px;
  background: #fff;
}
[data-theme="dark"] .site-layout-content {
  background: #141414;
}
</style>

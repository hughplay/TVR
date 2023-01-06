import Vue from 'vue';
import App from './App.vue';
import router from './router';
import VueTabulator from 'vue-tabulator';
import VModal from 'vue-js-modal';
import vSelect from 'vue-select'

import './assets/styles/index.css';
import 'vue-select/dist/vue-select.css';
import '@fortawesome/fontawesome-free/css/all.css';
import '@fortawesome/fontawesome-free/js/all.js';

Vue.use(VModal, {
  name: 'modal',
});

Vue.component("v-select", vSelect);

Vue.use(VueTabulator, {
  name: 'Vue-Tabulator',
});

Vue.config.productionTip = false;

new Vue({
  router,
  render: h => h(App),
}).$mount('#app');

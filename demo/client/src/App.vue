<template>
  <div id="app">
    <div class="shadow-md fixed top-0 bg-white w-full z-50">
      <div class="flex mx-auto max-w-5xl px-5 justify-between" style="height: 3.2em">
        <div class="flex flex-col justify-center">
          <router-link class="text-2xl font-bold text-blue-700" to="/"
            >TVR</router-link
          >
        </div>
        <div class="flex flex-row">
          <div
            class="border-b-2 border-transparent h-full mx-2 flex-col px-2
            flex justify-center"
            v-bind:class="{
              'border-blue-700': current_page.includes('dataset'),
              'text-blue-700': current_page.includes('dataset'),
            }"
          >
            <router-link
              class="hover:text-blue-700 trainsition-all duration-100"
              to="/dataset"
              >Dataset</router-link
            >
          </div>

          <div
            class="border-b-2 border-transparent h-full mx-2 flex-col px-2
            flex justify-center"
            v-bind:class="{
              'border-blue-700': current_page.includes('test'),
              'text-blue-700': current_page.includes('test'),
            }"
          >
            <router-link
              class="hover:text-blue-700 trainsition-all duration-100"
              to="/test"
              >Human Test</router-link
            >
          </div>
          <div
            class="border-b-2 border-transparent h-full flex-col px-2
             flex justify-center"
            v-bind:class="{
              'border-blue-700': current_page.includes('exp'),
              'text-blue-700': current_page.includes('exp'),
            }"
          >
            <router-link
              class="hover:text-blue-700 trainsition-all duration-100"
              to="/exp"
              >Experiments</router-link
            >
          </div>
        </div>
      </div>
    </div>
    <router-view class="mt-12" />
    <div class="bg-blue-900 py-10">
      <div class=" max-w-5xl mx-auto px-5 text-white font-normal">
        <p>
          <span class="text-lg"><i class="fas fa-envelope"/></span>
          <span class="ml-2">hongxin19b@ict.ac.cn</span>
        </p>
        <p>
          <span class="text-lg"><i class="fab fa-github"/></span>
          <span class="ml-2">hughplay</span>
        </p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  computed: {
    current_page: function() {
      return this.$route.path;
    },
  },
};
</script>

<style lang="scss">
@import '~vue-tabulator/dist/scss/bootstrap/tabulator_bootstrap4';
</style>

<style>
.divider {
  box-shadow: 0 1em 1em -1em rgba(0, 0, 0, 0.15);
}

.spinner {
  width: 35px;
  height: 35px;

  position: relative;
}
.double-bounce1,
.double-bounce2 {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background-color: #333;
  opacity: 0.6;
  position: absolute;
  top: 0;
  left: 0;

  -webkit-animation: sk-bounce 2s infinite ease-in-out;
  animation: sk-bounce 2s infinite ease-in-out;
}

.double-bounce2 {
  -webkit-animation-delay: -1s;
  animation-delay: -1s;
}

@-webkit-keyframes sk-bounce {
  0%,
  100% {
    -webkit-transform: scale(0);
  }
  50% {
    -webkit-transform: scale(1);
  }
}

@keyframes sk-bounce {
  0%,
  100% {
    transform: scale(0);
    -webkit-transform: scale(0);
  }
  50% {
    transform: scale(1);
    -webkit-transform: scale(1);
  }
}
</style>

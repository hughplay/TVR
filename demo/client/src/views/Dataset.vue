<template>
  <div>
    <div>
      <div class="max-w-5xl mx-auto px-5 pt-10">
        <div class="flex flex-row justify-between">
          <form class="flex-grow">
            <div class="flex items-center border-b border-b-2 border-blue-700 py-2">
              <p class="flex-shrink-0 font-semibold pr-3 text-lg">Data Root</p>
              <input
                class="appearance-none bg-transparent border-none w-full text-gray-700 mr-3 py-1 px-2 leading-tight focus:outline-none"
                v-model="datadir"
                type="text"
                list="default_datadirs"
              />
              <datalist id="default_datadirs">
                <option v-for="dir in default_datadirs" :value="dir" :key="dir">{{ dir }}</option>
              </datalist>
              <div
                class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 py-1 px-3 rounded shadow cursor-pointer transition-all duration-100"
                @click="get_info_list"
              >Explore</div>
            </div>
          </form>
          <div class="flex flex-row justify-between ml-10">
            <div class="items-center flex-shrink flex-row flex self-center text-sm">
              <div class="flex font-serif">
                <span class="border-b border-b-2 pb-1 border-blue-700 w-16">
                  <input
                    type="text"
                    class="appearance-none focus:outline-none w-16 bg-transparent text-center"
                    v-model="info_index"
                    id="info_index"
                  />
                </span>
                <span class="mx-2">/</span>
                <span class>{{ infolist.length }}</span>
              </div>
              <div
                class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 py-1 px-3 rounded shadow cursor-pointer transition-all duration-100 ml-5"
                @click="info_index -= 1"
              >Previous</div>
              <div
                class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 py-1 px-3 rounded shadow cursor-pointer transition-all duration-100 ml-5"
                @click="info_index += 1"
              >Next</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class>
      <div class="h-5 flex flex-row max-w-5xl mx-auto px-5 mt-5 justify-end">
        <div class="spinner self-end" v-show="loading">
          <div class="double-bounce1"></div>
          <div class="double-bounce2"></div>
        </div>
      </div>
    </div>

    <div class>
      <div class="flex flex-col pt-5 pb-10 max-w-5xl mx-auto px-5">
        <div class="flex flex-row justify-between items-end">
          <div class="flex flex-col justify-between">
            <div>
              <div class="border-b mb-5">
                <p class="text-lg font-semibold">Task Brief</p>
              </div>
              <p class="text-sm text-gray-700 mb-2">
                Given the initial state, the final state, and the description of objects in the initial state.
                The output should be the transformation(s) (
                <code>&lt;object, attribute, value&gt;</code>) that could achieve the shown change.
              </p>
              <p
                class="text-sm text-gray-700 mb-5"
              >Notice: Overlapping and moving out of the invisible area is not allowed throughout the whole transformation process.</p>
            </div>
            <div class="flex items-start">
              <div class="flex flex-col justify-center">
                <div
                  class="border-8 cursor-pointer border-white"
                  v-bind:class="{
                'shadow-xl': current_state == 0
                }"
                  @click="current_state = 0"
                >
                  <img :src="initial_src" alt class="state"/>
                </div>
                <div class="flex flex-row items-center justify-between mt-5">
                  <div class="mx-auto flex">
                    <p class="font-bold">Initial State</p>
                  </div>
                </div>
              </div>
              <div class="flex flex-col justify-center self-center pb-8">
                <svg
                  version="1.1"
                  id="Layer_1"
                  xmlns="http://www.w3.org/2000/svg"
                  xmlns:xlink="http://www.w3.org/1999/xlink"
                  x="0px"
                  y="0px"
                  width="100"
                  height="40"
                  viewBox="0 0 479.3 198.5"
                  enable-background="new 0 0 479.3 198.5"
                  xml:space="preserve"
                >
                  <g>
                    <line fill="#4D4D4D" x1="326.8" y1="95.1" x2="426" y2="95.1" />
                    <g>
                      <line
                        fill="none"
                        stroke="#474747"
                        stroke-width="20"
                        stroke-miterlimit="10"
                        x1="326.8"
                        y1="95.1"
                        x2="403.2"
                        y2="95.1"
                      />
                      <g>
                        <polygon
                          fill="#474747"
                          points="363.3,129 397.3,95.1 363.3,61.2 392.1,61.2 426,95.1 392.1,129 			"
                        />
                      </g>
                    </g>
                  </g>
                  <line
                    fill="#4D4D4D"
                    stroke="#474747"
                    stroke-width="20"
                    stroke-miterlimit="10"
                    x1="59"
                    y1="95.1"
                    x2="136.6"
                    y2="95.1"
                  />
                  <g>
                    <g>
                      <path
                        stroke="#000000"
                        stroke-miterlimit="10"
                        d="M268.7,63c0,4.5-1,8.5-3,12.1c-2,3.6-4.7,6.8-8,9.5c-3.2,2.6-6.8,5-10.7,7.1
			c-4,2.1-7.9,4-11.6,5.7l0.1,8.7H211V85.7c3.3-1.1,7.1-2.4,11.4-4c4.3-1.6,7.6-3,10-4.5c3.4-2.1,5.9-4.1,7.3-5.9
			c1.5-1.9,2.2-4.2,2.2-6.9c0-3.1-1-5.5-3-7.1c-2-1.6-4.6-2.8-7.7-3.6c-3.2-0.7-7.1-1.1-11.8-1.2c-4.7-0.1-10.1-0.2-16.1-0.2
			l3.2-20.1c1.1,0,3,0,5.7,0c2.7,0,5.2,0,7.6,0c5.8,0,11.1,0.4,15.7,1.1c4.6,0.7,9,1.9,13,3.5c6.2,2.4,11.2,5.9,14.8,10.2
			C266.9,51.4,268.7,56.8,268.7,63z M236.9,137.8h-26.4v-22.5h26.4V137.8z"
                      />
                    </g>
                  </g>
                </svg>
              </div>
              <div class="flex flex-col justify-center">
                <div
                  class="border-8 cursor-pointer border-white"
                  v-bind:class="{
                  'shadow-xl': current_state != 0
                }"
                  @click="current_state = n_state - 1"
                >
                  <img :src="final_src" alt class="state"/>
                </div>
                <div class="flex flex-col items-center justify-between mt-5">
                  <div class="flex mx-auto">
                    <p class="font-bold">Final State</p>
                  </div>
                  <div class="flex flex-row mt-3" v-show="Object.keys(final_images).length > 1">
                    <p
                      class="hover:text-black text-sm px-3 py-1 mx-1 shadow border rounded cursor-pointer transition-all duration-100"
                      v-show="'Camera_Left' in final_images"
                      v-bind:class="{
                    'text-black': view_final == 'Camera_Left',
                    'text-gray-500': view_final != 'Camera_Left',
                  }"
                      @click="view_final = 'Camera_Left'"
                    >Left</p>
                    <p
                      class="hover:text-black text-sm px-3 py-1 mx-1 shadow border rounded cursor-pointer transition-all duration-100"
                      v-show="'Camera_Center' in final_images"
                      v-bind:class="{
                    'text-black': view_final == 'Camera_Center',
                    'text-gray-500': view_final != 'Camera_Center',
                  }"
                      @click="view_final = 'Camera_Center'"
                    >Center</p>
                    <p
                      class="hover:text-black text-sm px-3 py-1 mx-1 shadow border rounded cursor-pointer transition-all duration-100"
                      v-show="'Camera_Right' in final_images"
                      v-bind:class="{
                    'text-black': view_final == 'Camera_Right',
                    'text-gray-500': view_final != 'Camera_Right',
                  }"
                      @click="view_final = 'Camera_Right'"
                    >Right</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="ml-8" style="min-width:350px">
            <table class="border-collapse text-sm">
              <thead class="border-b border-t">
                <tr>
                  <th class="font-semibold py-2 pl-4">Obj</th>
                  <th class="font-semibold py-2">Size</th>
                  <th class="font-semibold py-2">Color</th>
                  <th class="font-semibold py-2">Mat</th>
                  <th class="font-semibold py-2">Shape</th>
                  <th class="font-semibold py-2 pr-4">Position</th>
                </tr>
              </thead>
              <tbody class="border-b text-sm">
                <tr
                  v-for="(object, index) in this.get_state(this.current_state).objects"
                  :key="object.name"
                  class="text-gray-700"
                  v-bind:class="{'bg-gray-100': index % 2 == 0}"
                >
                  <td class="px-1 py-1 border-b text-center pl-4">{{ index }}</td>
                  <td class="px-1 py-1 border-b text-right">{{ object.size }}</td>
                  <td class="px-1 py-1 border-b text-right">{{ object.color }}</td>
                  <td class="px-1 py-1 border-b text-right">{{ object.material }}</td>
                  <td class="px-1 py-1 border-b text-right">{{ object.shape }}</td>
                  <td
                    class="px-1 border-b text-right pr-4"
                  >({{ object.position[0] }}, {{ object.position[1] }})</td>
                </tr>
              </tbody>
            </table>
            <div class="flex mt-5 flex-col text-center">
              <!-- <div class="w-3 h-5 border-l-4 border-blue-700 self-center"></div> -->
              <p class="font-semibold">The Initial State of Objects</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="bg-gray-100 shadow-inner">
      <div class="max-w-5xl mx-auto px-5 py-10">
        <div class="flex justify-between">
          <div class="flex flex-col justify-between">
            <div id="diagram" class="font-serif mx-auto"></div>
            <div class="border-b mb-5">
              <p class="text-lg font-semibold">Assistant Diagram</p>
            </div>
            <div class="flex flex-row justify-around">
              <p class="text-sm text-gray-700">
                This diagram is plotted according to the initial state of objects.
                It can help you understand the relative position of objects.
                Clicking the image of states can change the corresponding state of this diagram.
                This diagram can be downloaded in SVG or PNG format with the buttons right side.
              </p>
              <div class="flex flex-col ml-5">
                <p
                  class="my-1 flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 px-3 rounded shadow cursor-pointer transition-all duration-100"
                  @click="save_diagram('diagram', 'svg')"
                >SVG</p>
                <p
                  class="my-1 flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 px-3 rounded shadow cursor-pointer transition-all duration-100"
                  @click="save_diagram('diagram', 'png')"
                >PNG</p>
              </div>
            </div>
          </div>

          <div class="ml-5" style="min-width: 400px">
            <div class="border-b mb-5">
              <p class="text-lg font-semibold">Ground Truth</p>
            </div>
            <p class="text-sm text-gray-700">
              Transformations are represented as triplets, namely
              <code>&lt;object, attribute, value&gt;</code>.
              Be careful of the order. When you move objects, wrong order may lead to overlapping,
              which is not considered as a correct answer.
            </p>

            <table class="border-collapse mt-5 text-sm">
              <thead class="border-b border-t">
                <tr>
                  <th class="px-4 py-1"></th>
                  <th class="px-6 py-1">Object</th>
                  <th class="px-6 py-1">Attribute</th>
                  <th class="px-6 py-1">Value</th>
                </tr>
              </thead>
              <tbody class="border-b">
                <tr
                  v-for="(item, index) in ground_truth"
                  :key="item.obj_idx + item.pair"
                  class="border-b"
                >
                  <th class="py-1 px-4">{{ index + 1 }}</th>
                  <td class="py-1 px-6 text-center">{{ item.obj_idx }}</td>
                  <td class="py-1 px-6 text-center">{{ item.attr }}</td>
                  <td class="py-1 px-6 text-center">
                    {{
                    Array.isArray(item.target)
                    ? item.target[0] + ', ' + item.target[1]
                    : item.target
                    }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div class="py-10">
      <div class="flex flex-col max-w-5xl mx-auto px-5">
        <div class="flex flex-col">
          <div class="border-b mb-5">
            <p class="text-lg font-semibold">Detailed Information</p>
          </div>
        </div>
        <p class="mb-5 text-sm text-gray-700">Original information of this sample.</p>
        <TreeView :data="info" :options="{maxDepth: 1, rootObjectKey: 'sample_information'}"></TreeView>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import _ from 'lodash';

import { API } from '../js/api';
import { Diagram } from '../js/diagram';

import { TreeView } from 'vue-json-tree-view';

export default {
  components: {
    TreeView,
  },
  data() {
    return {
      default_datadirs: [],
      datadir: this.$route.query.datadir || '',
      infolist: [],
      info_index: this.$route.query.index || 0,
      current_info_index: 0,
      loading_list: false,
      loading_info: false,
      info: {},
      current_datadir: '',
      view_initial: 'Camera_Center',
      view_final: 'Camera_Center',
      current_state: 0,
      n_state: 0,
      diagram_scale: 4,
    };
  },
  mounted: function() {
    var self = this;
    axios.get(API.api_default_datadirs()).then(function(response) {
      self.default_datadirs = response.data.dirs;
      if (self.default_datadirs.length > 0) {
        self.datadir = self.datadir || self.default_datadirs[0];
        self.get_info_list();
      }
    });
  },
  methods: {
    save_diagram: Diagram.save_diagram,
    get_info_list: function() {
      if (!this.loading) {
        var self = this;
        this.loading_list = true;
        axios
          .get(API.api_infolist(this.datadir))
          .then(function(response) {
            self.infolist = response.data.infolist;
            self.current_datadir = self.datadir;
            self.get_info();
            self.view_initial = 'Camera_Center';
            self.view_final = 'Camera_Center';
          })
          .finally(() => (this.loading_list = false));
      }
    },

    get_info: _.debounce(function() {
      if (!this.loading_info) {
        var self = this;
        if (this.info_index < this.infolist.length) {
          this.loading_info = true;
          axios
            .get(API.api_info(this.infolist[this.info_index]))
            .then(function(response) {
              self.current_state = 0;
              self.info = response.data.info;
              self.current_info_index = self.info_index;
              Diagram.update_diagram(
                'diagram',
                self.current_objs,
                self.diagram_scale
              );
              self.n_state = self.info.states.length;
            })
            .finally(() => (this.loading_info = false));
        }
      }
    }, 500),

    get_state: function(state_idx) {
      if (!_.isUndefined(this.info.states)) {
        return this.info.states[state_idx];
      } else {
        return {};
      }
    },
  },
  watch: {
    info_index: function() {
      this.info_index = Number(this.info_index);
      if (this.info_index < this.infolist.length && this.info_index >= 0) {
        this.get_info();
      } else {
        this.info_index = this.current_info_index;
      }
    },
    current_state: function() {
      Diagram.update_diagram('diagram', this.current_objs, this.diagram_scale);
    },
  },
  computed: {
    initial_images: function() {
      if (!_.isUndefined(this.initial_state.images)) {
        return this.initial_state.images;
      } else {
        return {};
      }
    },
    final_images: function() {
      if (!_.isUndefined(this.final_state.images)) {
        return this.final_state.images;
      } else {
        return {};
      }
    },
    initial_src: function() {
      if (!_.isUndefined(this.initial_images[this.view_initial])) {
        return API.api_image(
          this.current_datadir,
          this.initial_images[this.view_initial]
        );
      } else {
        return '';
      }
    },
    final_src: function() {
      if (!_.isUndefined(this.final_images[this.view_final])) {
        return API.api_image(
          this.current_datadir,
          this.final_images[this.view_final]
        );
      } else {
        return '';
      }
    },
    ground_truth: function() {
      if (!_.isUndefined(this.info.transformations)) {
        return this.info.transformations;
      } else {
        return [];
      }
    },
    initial_state: function() {
      return this.get_state(0);
    },
    final_state: function() {
      return this.get_state(this.n_state - 1);
    },
    initial_objs: function() {
      return this.initial_state.objects;
    },
    final_objs: function() {
      return this.final_state.objects;
    },
    current_objs: function() {
      if (!_.isUndefined(this.info.states)) {
        return this.info.states[this.current_state].objects;
      } else {
        return [];
      }
    },
    loading: function() {
      return this.loading_list || this.loading_info;
    },
  },
};
</script>

<style scoped></style>

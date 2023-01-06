<template>
  <div>
    <div class="max-w-5xl mx-auto px-5 py-5">
      <div class="flex flex-row justify-between">
        <form class="flex-grow">
          <div class="flex items-center border-b border-b-2 border-blue-900 py-2">
            <p class="flex-shrink-0 font-bold pr-3">Config Root</p>
            <input
              class="appearance-none bg-transparent border-none w-full text-gray-700 mr-3 py-1 px-2 leading-tight focus:outline-none"
              v-model="configdir"
              type="text"
              placeholder="path to a configuration directory"
            />
            <div class="spinner flex-shrink-0 mx-5" v-show="loading">
              <div class="double-bounce1"></div>
              <div class="double-bounce2"></div>
            </div>
            <div
              class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 py-1 px-3 mt-1 rounded shadow cursor-pointer transition-all duration-100"
              @click="get_configs"
            >Explore</div>
          </div>
        </form>
      </div>
    </div>

    <div class="py-10">
      <div class="flex flex-col max-w-5xl mx-auto px-5">
        <div class="flex border-b">
          <p class="text-lg font-semibold">Basic</p>
        </div>
        <p class="text-sm py-5 text-gray-600">The experiment results of the Basic setting. Click a row to see the corresponding detail results.</p>
        <div class="flex">
          <Vue-Tabulator class="mt-2" v-model="results_basic" :options="options_basic" />
          <div class="flex-grow"></div>
        </div>
      </div>
    </div>

    <div class="py-10 shadow-inner bg-gray-100">
      <div class="flex flex-col max-w-5xl mx-auto px-5">
        <div class="flex border-b">
          <p class="text-lg font-semibold">View</p>
        </div>
        <p class="text-sm py-5 text-gray-600">The experiment results of the View setting. Click a row to see the corresponding detail results.</p>
        <Vue-Tabulator class="mt-2" v-model="results_view" :options="options_view" />
      </div>
    </div>

    <div class="py-10">
      <div class="flex flex-col max-w-5xl mx-auto px-5">
        <div class="flex border-b">
          <p class="text-lg font-semibold">Event</p>
        </div>
        <p class="text-sm py-5 text-gray-600">The experiment results of the Event setting. Click a row to see the corresponding detail results.</p>
        <Vue-Tabulator class="mt-2" v-model="results_event" :options="options_event" />
      </div>
    </div>

    <div class="py-10 shadow-inner bg-gray-100">
      <div class="flex flex-col max-w-5xl mx-auto px-5">
        <div class="flex border-b">
          <p class="text-lg font-semibold">Event (Position Only)</p>
        </div>
        <p class="text-sm py-5 text-gray-600">The experiment results of the Event (position only) setting. Click a row to see the corresponding detail results.</p>
        <Vue-Tabulator class="mt-2" v-model="results_event_sp" :options="options_event" />
      </div>
    </div>

    <div class="mt-10"></div>
    <modal
      name="detail"
      classes="bg-white relative border-l-2 border-blue-700 px-10 py-8"
      @opened="modal_opened"
      :width="window_width - 100"
      :adaptive="true"
      height="100%"
      :pivotX="1"
    >
      <div class="absolute top-1/4 left-1/2" v-show="loading">
        <div class="spinner">
          <div class="double-bounce1"></div>
          <div class="double-bounce2"></div>
        </div>
      </div>
      <div
        @click="hide_modal"
        class="cursor-pointer absolute right-0 top-0 px-2 m-5 text-2xl text-blue-900"
      >
        <i class="fas fa-times" />
      </div>
      <div class>
        <div>
          <div class="flex border-b items-center">
            <p class="text-lg font-semibold">Sample Detail</p>
            <div
              class="flex-shrink-0 text-blue-800 text-sm px-3 cursor-pointer ml-5"
              @click="sample_visible = !sample_visible"
            >{{ sample_visible ? "Hide" : "Show" }}</div>
          </div>
          <div v-show="sample_visible">
            <p
              class="text-sm py-5"
            >Click a row in the table below to check the detail information of a sample. (Current ID: {{current_id}})</p>
            <div class="flex mt-2">
              <div>
                <div>
                  <div class="flex flex-row items-center justify-between mb-1">
                    <div class="mx-auto flex">
                      <p class="font-bold">Initial State</p>
                    </div>
                  </div>
                  <div
                    class="border-4 cursor-pointer border-white"
                    v-bind:class="{
                'shadow-xl': current_state == 0
                }"
                    @click="current_state = 0"
                  >
                    <img :src="initial_src" style="width:160px;height:120px" />
                  </div>
                </div>
                <div class="mt-2">
                  <div class="flex flex-row items-center justify-between mb-1">
                    <div class="mx-auto flex">
                      <p class="font-bold">Final State ({{current_final_view}})</p>
                    </div>
                  </div>
                  <div
                    class="border-4 cursor-pointer border-white"
                    v-bind:class="{
                  'shadow-xl': current_state != 0
                }"
                    @click="current_state = n_state - 1"
                  >
                    <img :src="final_src" style="width:160px;height:120px" />
                  </div>
                </div>
              </div>
              <div class="ml-5" style="min-width:350px">
                <div class="flex flex-col text-center mb-2">
                  <!-- <div class="w-3 h-5 border-l-4 border-blue-700 self-center"></div> -->
                  <p class="font-semibold">The Initial State of Objects</p>
                </div>
                <table class="border-collapse text-sm">
                  <thead class="border-b border-t">
                    <tr>
                      <th class="font-semibold py-1 pl-4">Obj</th>
                      <th class="font-semibold py-1">Size</th>
                      <th class="font-semibold py-1">Color</th>
                      <th class="font-semibold py-1">Mat</th>
                      <th class="font-semibold py-1">Shape</th>
                      <th class="font-semibold py-1 pr-4">Position</th>
                    </tr>
                  </thead>
                  <tbody class="border-b text-sm">
                    <tr
                      v-for="(object, index) in current_objects"
                      :key="object.name"
                      class="text-gray-700"
                      v-bind:class="{'bg-gray-100': index % 2 == 0}"
                    >
                      <td class="px-1 border-b text-center pl-4">{{ index }}</td>
                      <td class="px-1 border-b text-right">{{ object.size }}</td>
                      <td class="px-1 border-b text-right">{{ object.color }}</td>
                      <td class="px-1 border-b text-right">{{ object.material }}</td>
                      <td class="px-1 border-b text-right">{{ object.shape }}</td>
                      <td
                        class="px-1 border-b text-right pr-4"
                      >({{ object.position[0] }}, {{ object.position[1] }})</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div>
                <div id="diagram"></div>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div class="flex border-b mt-5">
            <p class="text-lg font-semibold">Test Results</p>
          </div>
          <p class="text-sm py-5">
            This table shows the prediction of {{ detail_model }} in the {{ detail_type }} setting.
            Click one row to see corresponding inputs.
          </p>
          <div id="detail" class=""></div>
        </div>
      </div>
    </modal>
  </div>
</template>

<script>
import axios from 'axios';
import _ from 'lodash';
import Tabulator from 'tabulator-tables';
import { API } from '../js/api';
import { Diagram } from '../js/diagram';

export default {
  data() {
    return {
      configdir: '',
      current_configdir: '',
      current_id: 0,
      loading_configs: false,
      loading_result: false,
      loading_list: false,
      configs: {},
      test_result: {},
      detail_type: '',
      detail_model: '',
      detail_table: null,
      current_test_sample: {},
      current_final_view: '',
      diagram_scale: 3.5,
      current_state: 0,
      n_state: 0,
      sample_visible: true,
      options_basic: {
        columns: [
          { title: 'Model', field: 'name', headerSort: false },
          {
            title: 'ObjAcc',
            field: 'acc_obj',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            align: 'center',
          },
          {
            title: 'AttrAcc',
            field: 'acc_attr',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
          {
            title: 'ValAcc',
            field: 'acc_pair',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
          {
            title: 'Acc',
            field: 'acc',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
        ],
        initialSort: [{ column: 'acc', dir: 'desc' }],
        rowClick: this.get_detail,
      },
      options_view: {
        dataTree: true,
        dataTreeStartExpanded: true,
        columns: [
          { title: 'Model', field: 'name', headerSort: false },
          {
            title: 'ObjAcc',
            field: 'acc_obj',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
          {
            title: 'AttrAcc',
            field: 'acc_attr',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
          {
            title: 'ValAcc',
            field: 'acc_pair',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
          {
            title: 'Acc',
            field: 'acc',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
        ],
        initialSort: [{ column: 'acc', dir: 'desc' }],
        initialFilter: [{ field: 'acc_attr', type: '>', value: 0 }],
        rowClick: this.get_detail,
      },
      options_event: {
        layout: 'fitData',
        dataTree: true,
        dataTreeStartExpanded: true,
        columns: [
          { title: 'Model', field: 'name', headerSort: false },
          {
            title: 'AD',
            field: 'avg_dist',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
          {
            title: 'AND',
            field: 'avg_norm_dist',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
          {
            title: 'LAcc',
            field: 'loose_acc',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
          {
            title: 'Acc',
            field: 'acc',
            sorter: 'number',
            headerSort: false,
            formatter: this.number_formatter,
            cssClass: 'alignRight',
            align: 'right',
          },
        ],
        initialSort: [{ column: 'acc', dir: 'desc' }],
        rowClick: this.get_detail,
      },
      options_detail_basic: {
        layout: 'fitData',
        height: '100%',
        rowClick: this.show_image,
        columns: [
          { title: 'ID', field: 'id' },
          {
            title: 'Obj',
            field: 'obj_pred',
            headerSortStartingDir: 'asc',
            formatter: this.obj_formatter,
            align: 'left',
          },
          {
            title: 'Attr',
            field: 'attr_pred',
            headerSortStartingDir: 'asc',
            formatter: this.attr_formatter,
            align: 'left',
          },
          {
            title: 'Val',
            field: 'pair_pred',
            headerSortStartingDir: 'asc',
            formatter: this.val_formatter,
            align: 'left',
          },
          {
            title: 'Overall',
            field: 'correct',
            headerSortStartingDir: 'asc',
            formatter: this.bool_formatter,
            align: 'center',
          },
        ],
      },
      options_detail_view: {
        layout: 'fitData',
        height: '100%',
        rowClick: this.show_image,
        columns: [
          { title: 'ID', field: 'id' },
          {
            title: 'View',
            field: 'view',
            headerSortStartingDir: 'asc',
            formatter: this.view_formatter,
            align: 'left',
          },
          {
            title: 'Obj',
            field: 'obj_pred',
            headerSort: false,
            formatter: this.obj_formatter,
            align: 'left',
          },
          {
            title: 'Attr',
            field: 'attr_pred',
            headerSortStartingDir: 'asc',
            formatter: this.attr_formatter,
            align: 'left',
          },
          {
            title: 'Val',
            field: 'pair_pred',
            headerSortStartingDir: 'asc',
            formatter: this.val_formatter,
            align: 'left',
          },
          {
            title: 'Overall',
            field: 'correct',
            headerSortStartingDir: 'asc',
            formatter: this.bool_formatter,
            align: 'center',
          },
        ],
      },
      options_detail_event: {
        layout: 'fitData',
        height: '100%',
        rowClick: this.show_image,
        columns: [
          { title: 'ID', field: 'id' },
          {
            title: 'Prediction',
            field: 'pred',
            headerSort: false,
            formatter: this.transformation_formatter,
            align: 'left',
          },
          {
            title: 'Ground Truth',
            field: 'target',
            headerSort: false,
            formatter: this.transformation_formatter,
            align: 'left',
          },
          {
            title: 'Distance',
            field: 'dist',
            sorter: 'number',
            headerSortStartingDir: 'asc',
            align: 'center',
          },
          {
            title: 'No Overlapping',
            field: 'err_overlap',
            headerSortStartingDir: 'asc',
            formatter: this.reverse_bool_formatter,
            align: 'center',
          },
          {
            title: 'Valid Position',
            field: 'err_invalid_position',
            headerSortStartingDir: 'asc',
            formatter: this.reverse_bool_formatter,
            align: 'center',
          },
          {
            title: 'Overall',
            field: 'correct',
            headerSortStartingDir: 'asc',
            formatter: this.bool_formatter,
            align: 'center',
          },
        ],
      },
    };
  },
  mounted: function() {
    var self = this;
    axios.get(API.api_default_configdir()).then(function(response) {
      self.configdir = response.data.dir;
      if (response.data.exists) {
        self.get_configs();
      }
    });
  },
  methods: {
    get_configs: function() {
      if (!this.loading) {
        var self = this;
        this.loading_configs = true;
        axios
          .get(API.api_configs(this.configdir))
          .then(function(response) {
            self.configs = response.data;
            self.current_configdir = self.configdir;
          })
          .finally(() => (this.loading_configs = false));
      }
    },

    number_formatter: function(cell) {
      return Number.parseFloat(cell.getValue()).toFixed(4);
    },
    bool_formatter: function(cell) {
      if (cell.getValue()) {
        return '<i class="text-green-500 fas fa-check"></i>';
      } else {
        return '<i class="text-red-500 fas fa-times"></i>';
      }
    },
    reverse_bool_formatter: function(cell) {
      if (cell.getValue()) {
        return '<i class="text-red-500 fas fa-times"></i>';
      } else {
        return '<i class="text-green-500 fas fa-check"></i>';
      }
    },
    obj_formatter: function(cell) {
      let row = cell.getRow().getData();
      if (row.obj_pred == row.obj_target) {
        return (
          '<i class="text-green-500 fas fa-check"></i> ' +
          '(' +
          row.obj_target +
          ')'
        );
      } else {
        return (
          '<i class="text-red-500 fas fa-times"></i> ' +
          '(' +
          row.obj_pred +
          ' &#8594; ' +
          row.obj_target +
          ')'
        );
      }
    },
    attr_formatter: function(cell) {
      let row = cell.getRow().getData();
      if (row.attr_pred == row.attr_target) {
        return (
          '<i class="text-green-500 fas fa-check"></i> ' +
          '(' +
          this.test_result.attrs[row.attr_target] +
          ')'
        );
      } else {
        return (
          '<i class="text-red-500 fas fa-times"></i> ' +
          '(' +
          this.test_result.attrs[row.attr_target] +
          '<br/>&#8594; ' +
          this.test_result.attrs[row.attr_pred] +
          ')'
        );
      }
    },
    val_formatter: function(cell) {
      let row = cell.getRow().getData();
      if (row.pair_pred == row.pair_target) {
        return (
          '<i class="text-green-500 fas fa-check"></i> ' +
          '(' +
          this.test_result.pairs[row.pair_target] +
          ')'
        );
      } else {
        return (
          (row.correct
            ? '<i class="text-green-500 fas fa-times"></i> '
            : '<i class="text-red-500 fas fa-times"></i> ') +
          '(' +
          this.test_result.pairs[row.pair_target] +
          '<br/>&#8594; ' +
          this.test_result.pairs[row.pair_pred] +
          ')'
        );
      }
    },
    view_formatter: function(cell) {
      return this.test_result.final_views[cell.getValue()];
    },
    transformation_formatter: function(cell) {
      var res = '';
      for (var item of cell.getValue()) {
        res += item[0] + ', ' + this.test_result.pairs[item[1]] + '<br/>';
      }
      return res;
    },

    get_detail: function(e, row) {
      if (!this.loading) {
        var self = this;
        this.loading_result = true;
        this.detail_model = row.getData().model;
        this.current_test_sample = {};
        this.current_final_view = '';
        self.$modal.show('detail');
        axios
          .get(API.api_result(row.getData().config))
          .then(function(response) {
            self.test_result = response.data;
            self.detail_type = self.test_result['datatype'];
            self.options_detail.data = self.results_detail;
            self.detail_table = new Tabulator('#detail', self.options_detail);
            self.detail_table.setHeight(window.innerHeight - 520);
            Diagram.update_diagram(
              'diagram',
              self.current_objects,
              self.diagram_scale
            );
          })
          .finally(() => (this.loading_result = false));
      }
    },
    show_image: function(e, row) {
      if (!this.loading) {
        var data = row.getData();
        this.current_final_view = data.view
          ? this.test_result.final_views[data.view]
          : 'center';
        var configpath = this.test_result.configpath;
        var self = this;
        this.loading_test_image = true;
        this.current_id = data.id;
        axios
          .get(API.api_test_sample(configpath, data.id))
          .then(function(response) {
            self.current_test_sample = response.data;
            self.n_state = self.current_test_sample.info.states.length;
          })
          .finally(() => (this.loading_test_image = false));
      }
    },

    show_modal: function() {
      this.$modal.show('detail');
    },
    hide_modal: function() {
      this.$modal.hide('detail');
    },

    modal_opened: function() {},

    get_configs_by_datatype: function(datatype) {
      var res = [];
      if (typeof this.configs.configs != 'undefined') {
        for (var config of this.configs.configs) {
          if (config.data == datatype) {
            res.push(config);
          }
        }
      }
      return res;
    },
    get_results_by_datatype: function(datatype) {
      var results = [];
      if (datatype == 'event.sp') {
        datatype = 'event_sp';
      }
      for (var config of this['configs_' + datatype]) {
        if (config.result && config.result.version == config.version) {
          var result = config.result;
          result.model = config.model;
          result.name = _.replace(config.version, config.data + '.', '');
          result.config = config.config;
          if (datatype == 'view') {
            result._children = [];
            for (var view of ['center', 'left', 'right']) {
              var view_detail = {};
              view_detail.model = config.model;
              view_detail.config = config.config;
              for (var metric of ['acc_obj', 'acc_attr', 'acc_pair', 'acc']) {
                var name = view + '_' + metric;
                result[name] && (view_detail[metric] = result[name]);
              }
              if (Object.keys(view_detail).length > 0) {
                view_detail.name = 'center -> ' + view;
                result._children.push(view_detail);
              }
            }
          } else if (datatype == 'event' || datatype == 'event_sp') {
            result._children = [];
            for (var step of _.keys(result.step_result)) {
              var r = result.step_result[step]
              r.name = 'step_' + step;
              r.model = config.model;
              r.config = config.config;
              result._children.push(r);
            }
          }
          results.push(result);
        }
      }
      return results;
    },
  },
  watch: {
    current_objects: function() {
      Diagram.update_diagram(
        'diagram',
        this.current_objects,
        this.diagram_scale
      );
    },
    sample_visible: function() {
      if (this.sample_visible) {
        this.detail_table.setHeight(window.innerHeight - 520);
      } else {
        this.detail_table.setHeight(window.innerHeight - 200);
      }
    },
  },
  computed: {
    window_width: function() {
      return window.innerWidth;
    },
    loading: function() {
      return (
        this.loading_configs || this.loading_result || this.loading_test_image
      );
    },

    configs_basic: function() {
      return this.get_configs_by_datatype('basic');
    },
    configs_view: function() {
      return this.get_configs_by_datatype('view');
    },
    configs_event: function() {
      return this.get_configs_by_datatype('event');
    },
    configs_event_sp: function() {
      return this.get_configs_by_datatype('event.sp');
    },

    results_basic: function() {
      return this.get_results_by_datatype('basic');
    },
    results_view: function() {
      return this.get_results_by_datatype('view');
    },
    results_event: function() {
      return this.get_results_by_datatype('event');
    },
    results_event_sp: function() {
      return this.get_results_by_datatype('event.sp');
    },
    options_detail: function() {
      if (['basic', 'view', 'event', 'event.sp'].includes(this.detail_type)) {
        var _type = this.detail_type == 'event.sp' ? 'event' : this.detail_type;
        return this['options_detail_' + _type];
      } else {
        return {};
      }
    },
    results_detail: function() {
      var results = [];
      this.test_result.detail && (results = this.test_result.detail);
      return results;
    },
    current_objects: function() {
      if (this.current_test_sample.info) {
        return this.current_test_sample.info.states[this.current_state].objects;
      } else {
        return [];
      }
    },
    initial_src: function() {
      if (
        this.current_test_sample.info &&
        this.current_test_sample.info.states
      ) {
        var states = this.current_test_sample.info.states;
        return API.api_h5image(
          this.current_test_sample.configpath,
          states[0].images['Camera_Center']
        );
      } else {
        return '';
      }
    },
    final_src: function() {
      if (
        this.current_test_sample.info &&
        this.current_test_sample.info.states
      ) {
        var states = this.current_test_sample.info.states;
        return API.api_h5image(
          this.current_test_sample.configpath,
          states[states.length - 1].images[
            'Camera_' +
              this.current_final_view.charAt(0).toUpperCase() +
              this.current_final_view.slice(1)
          ]
        );
      } else {
        return '';
      }
    },
  },
};
</script>

<style>
.tabulator {
  display: inline-block !important;
  font-size: 14px;
}
.tabulator .tabulator-header {
  background: transparent;
  border: 0;
  width: auto;
  display: tabale;
  font-weight: 600;
}

.tabulator .tabulator-headers {
  width: auto;
  display: table;
  border-color: #e2e8f0;
  color: #2c5282;
  border-top-width: 0px;
  border-bottom-width: 1px;
}

.tabulator .tabulator-header .tabulator-col {
  background: transparent;
  height: auto !important;
}

.tabulator-col-content,
.tabulator-row .tabulator-cell {
  padding: 0.5em;
  height: auto !important;
}

.tabulator-row {
  border-bottom: 1px solid #e2e8f0;
  min-height: auto !important;
}

.tabulator-tableHolder {
  display: inline-block !important;
  width: auto !important;
  border-color: #e2e8f0;
  border-bottom-width: 1px;
  overflow: -moz-scrollbars-none;
  -ms-overflow-style: none;
}
.tabulator-tableHolder::-webkit-scrollbar {
  width: 0 !important;
}

.tabulator-cell {
  padding: 0.5em 1em !important;
}

.tabulator-header .alignRight {
  text-align: right !important;
}
</style>

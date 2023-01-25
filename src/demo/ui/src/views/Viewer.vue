<template>
  <div class="flex flex-col 2xl:flex-row gap-5">
    <!-- Left Panel -->
    <div class="grow overflow-auto flex flex-col gap-5" style="height: 85vh">
      <!-- Data Selector -->
      <div>
        <a-card title="Data">
          <div class="flex flex-col gap-5">
            <!-- <div class="flex flex-col gap-2">
              <a-input v-model:value="data_root" addon-before="Data Root" />
            </div> -->
            <div>
              <a-tabs v-model:activeKey="mode">
                <!-- Load from dataset -->
                <a-tab-pane key="dataset" tab="Dataset">
                  <div class="flex flex-col gap-2">
                    <div class="flex gap-2 flex-row">
                      <a-radio-group v-model:value="problem">
                        <a-radio-button value="basic">Basic</a-radio-button>
                        <a-radio-button value="event"
                          >Event / View</a-radio-button
                        >
                      </a-radio-group>
                    </div>
                    <div class="">
                      <a-radio-group v-model:value="data_split">
                        <a-radio-button value="train">Train</a-radio-button>
                        <a-radio-button value="val">Val</a-radio-button>
                        <a-radio-button value="test">Test</a-radio-button>
                      </a-radio-group>
                    </div>
                  </div>
                </a-tab-pane>
                <!-- Load from experiment -->
                <a-tab-pane key="exp" tab="Experiment" force-render>
                  <div class="flex flex-col gap-2">
                    <div class="flex">
                      <a-select
                        v-model:value="selected_exps"
                        class="w-full"
                        mode="multiple"
                        :options="exps"
                        :loading="loading_exps"
                      ></a-select>
                      <a-button @click="load_explist" class="ml-2">
                        Refresh
                      </a-button>
                    </div>
                    <div>
                      <a-tag color="blue" v-if="this.exp_problem">
                        {{ this.exp_problem }}
                      </a-tag>
                    </div>
                  </div>
                </a-tab-pane>
              </a-tabs>
            </div>

            <div class="flex flex-row gap-2">
              <a-input-number id="inputNumber" v-model:value="sample_idx" />
              <a-button
                type="primary"
                ghost
                @click="load_sample"
                :loading="loading"
              >
                Load
              </a-button>
            </div>
          </div>
        </a-card>
      </div>

      <!-- States -->
      <div class="flex gap-4">
        <!-- Initial State (Center) -->
        <a-card @click="current_state = 0" class="cursor-pointer grow">
          <template #title>
            <div :class="{ 'text-blue-500': current_state == 0 }">
              Initial State
            </div>
          </template>
          <!-- :src="this.config.states[0].images['Camera_Center']" -->
          <img
            :src="img_url(this.config.states[0].images['Camera_Center'])"
            class="mx-auto"
          />
          <div class="text-center mt-5 text-gray-500">Center</div>
        </a-card>

        <!-- Final State -->
        <a-card
          @click="current_state = this.config.states.length - 1"
          class="cursor-pointer"
        >
          <template #title>
            <div :class="{ 'text-blue-500': current_state == 1 }">
              Final State
            </div>
          </template>
          <!-- Left -->
          <a-card-grid class="w-1/3" :hoverable="false">
            <img
              :src="
                img_url(
                  this.config.states[this.config.states.length - 1].images[
                    'Camera_Left'
                  ]
                )
              "
            />
            <div class="text-center mt-5 text-gray-500">Left</div>
          </a-card-grid>

          <!-- Center -->
          <a-card-grid class="w-1/3" :hoverable="false">
            <img
              :src="
                img_url(
                  this.config.states[this.config.states.length - 1].images[
                    'Camera_Center'
                  ]
                )
              "
            />
            <div class="text-center mt-5 text-gray-500">Center</div>
          </a-card-grid>

          <!-- Right -->
          <a-card-grid class="w-1/3" :hoverable="false">
            <img
              :src="
                img_url(
                  this.config.states[this.config.states.length - 1].images[
                    'Camera_Right'
                  ]
                )
              "
            />
            <div class="text-center mt-5 text-gray-500">Right</div>
          </a-card-grid>
        </a-card>
      </div>

      <!-- Diagram -->
      <div class="mt-5 flex gap-4">
        <a-card title="Illustration">
          <div class="flex flex-col items-center">
            <div id="diagram" class="font-serif"></div>
            <div class="flex gap-2 pt-5">
              <a-button @click="save_diagram('diagram', 'svg')">
                <div class="flex items-center gap-2">
                  <download-outlined />
                  <div>SVG</div>
                </div>
              </a-button>
              <a-button @click="save_diagram('diagram', 'png')">
                <div class="flex items-center gap-2">
                  <download-outlined />
                  <div>PNG</div>
                </div>
              </a-button>
            </div>
          </div>
        </a-card>

        <!-- Transformation -->
        <a-card title="Transformation" class="grow">
          <a-timeline>
            <a-timeline-item
              v-for="(t, i) in this.config.transformations"
              :key="this.config.idx + '.' + i"
            >
              <div class="flex gap-2">
                <div class="w-6 text-center border rounded-full">
                  {{ t.obj_idx }}
                </div>
                <div>{{ t.attr }}</div>
                <div>{{ t.val }}</div>
              </div>
            </a-timeline-item>
          </a-timeline>
          <div class="text-gray-500">
            {{ this.config.transformations.length }} step
            {{ this.config.transformations.length > 1 ? "s" : "" }}
          </div>
        </a-card>
      </div>

      <!-- Preds -->
      <a-card title="Experiment Results" v-if="preds.length > 0">
        <a-tabs v-model:activeKey="visible_exp">
          <a-tab-pane :key="exp.name" :tab="exp.name" v-for="exp in preds">
            <div class="my-2">
              <a-timeline>
                <a-timeline-item
                  v-for="(t, i) in exp.pred"
                  :key="this.config.idx + '.' + exp.name + '.' + i"
                >
                  <div class="flex gap-2">
                    <div class="w-6 text-center border rounded-full">
                      {{ t.obj }}
                    </div>
                    <div>{{ t.attr }}</div>
                    <div>{{ t.val }}</div>
                  </div>
                </a-timeline-item>
              </a-timeline>

              <!-- tags -->
              <div class="flex gap-2">
                <a-tag v-if="exp.correct" color="green">correct</a-tag>
                <a-tag v-else color="red">wrong</a-tag>
                <a-tag v-if="exp.loose_correct" color="green"
                  >loose correct</a-tag
                >
                <a-tag v-if="!exp.correct" color="orange"
                  >dist: {{ exp.dist }}</a-tag
                >
                <a-tag v-if="exp.err_overlap" color="orange">overlap</a-tag>
              </div>
              <div></div>
            </div>
          </a-tab-pane>
        </a-tabs>
      </a-card>

      <!-- overall metrics -->
      <a-card title="Metrics" v-if="metrics_table.columns">
        <a-table
          :scroll="{ x: true }"
          :dataSource="metrics_table.dataSource"
          :columns="metrics_table.columns"
        />
      </a-card>
    </div>

    <!-- Editor on the right -->
    <a-card
      title="Sample Information"
      class="overflow-auto"
      style="height: 85vh; min-width: 48em"
    >
      <div class="flex justify-end">
        <a-button class="" type="primary" ghost @click="submit_config"
          >Show</a-button
        >
      </div>
      <div id="editor" class="w-full mt-2"></div>
    </a-card>
  </div>
</template>

<script>
import { EditorState } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { defaultKeymap } from "@codemirror/commands";
import {
  syntaxHighlighting,
  defaultHighlightStyle,
} from "@codemirror/language";
import { json } from "@codemirror/lang-json";
import _ from "lodash";
import { message } from "ant-design-vue";
import "ant-design-vue/es/message/style/css";
import { DownloadOutlined } from "@ant-design/icons-vue";

import example from "../assets/example.json";
import { Diagram } from "../js/diagram";

import { API } from "../js/api";
import axios from "axios";

export default {
  name: "Viewer",

  components: {
    DownloadOutlined,
  },

  data() {
    return {
      config: example,
      editor: {},
      full_config: {},
      checked: {},
      update_when_checked: true,

      diagram_scale: 4,
      current_state: 0,

      mode: "dataset",
      problem: "event",
      data_split: "test",
      sample_idx: 0,
      exp_root: "/log/exp/tvr",
      exps: [],
      selected_exps: [],
      preds: [],
      exp_problem: "",
      visible_exp: "",
      metrics_table: {},

      loading: false,
      loading_exps: false,
    };
  },

  mounted() {
    this.load_explist();
    this.create_view(this.config);
    this.update_diagram();
  },

  methods: {
    gen_metrics_table(metrics) {
      let dataSource = [];
      let columns = [{
          title: "Exp",
          dataIndex: "exp",
          key: "exp",
          fixed: true,
      }];
      let concern_metrics = ["acc", "dist", "diff"];

      // iterate over metrics (array) with key and index
      for (let [i, exp] of metrics.entries()) {
        let row = {
          key: i,
        };
        row["exp"] = exp["name"];
        for (let metric in exp) {
          // if metric name includes concern metrics
          if (
            concern_metrics.some(
              (m) => metric.includes(m) && metric.startsWith("test")
            )
          ) {
            // reserve four decimal places
            row[metric] = exp[metric].toFixed(4);

            if (i == 0) {
              // title example: test/loose_acc -> Loose Acc
              let title = metric
                .split("/")
                .slice(1)
                .join("/")
                .split("_")
                .map((w) => w[0].toUpperCase() + w.slice(1))
                .join(" ");

              columns.push({
                title: title,
                dataIndex: metric,
                key: metric,
              });
            }
          }
        }
        dataSource.push(row);
      }

      let table = {
        dataSource: dataSource,
        columns: columns,
      };

      console.log(table);

      return table;
    },

    load_explist() {
      this.loading_exps = true;
      axios(API.get_explist())
        .then((res) => {
          if (res.data.error) {
            message.error(res.data.error);
          } else {
            let exps = [];
            for (let exp of res.data) {
              exps.push({
                label: exp,
                value: exp,
              });
            }
            this.exps = exps;
          }
        })
        .finally(() => {
          this.loading_exps = false;
        });
    },

    load_sample() {
      this.loading = true;

      // load a single sample
      if (this.mode == "dataset") {
        axios(API.get_sample(this.problem, this.data_split, this.sample_idx))
          .then((res) => {
            if (res.data.error) {
              message.error(res.data.error);
            } else {
              this.config = res.data;
              this.update_view(this.config);
            }
          })
          .finally(() => {
            this.loading = false;
          });
      }
      // load experiment results and the corresponding sample
      else {
        let selected_explist = [];
        // transform the object to a list
        for (let exp of this.selected_exps) {
          selected_explist.push(exp);
        }
        axios(API.get_exps(selected_explist, this.sample_idx))
          .then((res) => {
            if (res.data.error) {
              message.error(res.data.error);
            } else {
              this.config = res.data.gt;
              this.preds = res.data.preds;
              this.exp_problem = res.data.problem;
              this.visible_exp = this.preds[0].name;
              this.update_view(this.config);
              this.metrics_table = this.gen_metrics_table(res.data.metrics);
            }
          })
          .finally(() => {
            this.loading = false;
          });
      }
    },

    img_url(name) {
      if (name.startsWith("/")) {
        return name;
      } else {
        return API.get_img_url(name);
      }
    },

    save_diagram: Diagram.save_diagram,

    update_diagram() {
      Diagram.update_diagram("diagram", this.current_objs, this.diagram_scale);
    },

    config2text(config, renew) {
      if (renew) {
        this.config = JSON.parse(JSON.stringify(config));
      }
      let text = JSON.stringify(config, null, 2);
      return text;
    },

    create_view(config) {
      let start_state = EditorState.create({
        doc: this.config2text(config, true),
        extensions: [
          keymap.of(defaultKeymap),
          syntaxHighlighting(defaultHighlightStyle),
          json(),
        ],
      });
      this.editor = new EditorView({
        state: start_state,
        parent: document.querySelector("#editor"),
      });
    },

    update_view(config) {
      this.editor.dispatch({
        changes: {
          from: 0,
          to: this.editor.state.doc.length,
          insert: this.config2text(config, true),
        },
      });
    },

    submit_config() {
      this.submitting_requests = true;
      let current_config = JSON.parse(this.editor.state.doc.toString());
      this.config = current_config;
      this.update_view(this.config);
    },
  },

  computed: {
    current_objs() {
      if (!_.isUndefined(this.config.states)) {
        return this.config.states[this.current_state].objects;
      } else {
        return [];
      }
    },
  },

  watch: {
    current_state: function () {
      Diagram.update_diagram("diagram", this.current_objs, this.diagram_scale);
    },
    config: function () {
      this.current_state = 0;
      Diagram.update_diagram("diagram", this.current_objs, this.diagram_scale);
    },
    sample_idx: function () {
      this.load_sample();
    },
    problem: function () {
      this.load_sample();
    },
    split: function () {
      this.load_sample();
    },
  },
};
</script>

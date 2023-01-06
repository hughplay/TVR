<template>
  <div>
    <div>
      <div class="max-w-5xl mx-auto px-5 pt-10">
        <div class="border-b mb-5">
          <p class="text-lg font-semibold">Setting</p>
        </div>
        <p
          class="text-sm text-gray-600 mb-2"
        >Fill in the user name, select a problem, and then start to test.</p>
        <div class="flex flex-col">
          <div class="flex">
            <div class="flex mr-5 items-center">
              <p class="font-semibold pr-3 text-sm text-gray-600">User Name</p>
              <input
                class="w-32 appearance-none bg-transparent border-b-2 border-blue-700 text-gray-700 mr-3 py-1 px-2 leading-tight focus:outline-none"
                v-model="input_user"
                type="text"
              />
            </div>
            <div class="flex items-center">
              <p class="font-semibold text-sm text-gray-600 mr-3">Problem</p>
              <div class="relative">
                <select
                  class="block appearance-none w-full border-2 border-blue-700 text-gray-700 py-1 px-4 pr-8 rounded leading-tight focus:outline-none focus:bg-white focus:border-gray-500"
                  v-model="input_problem"
                  name="problem"
                  id="problem"
                >
                  <option v-for="p in problems" :value="p" :key="p">{{ p }}</option>
                </select>
                <div
                  class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700"
                >
                  <svg
                    class="fill-current h-4 w-4"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                  >
                    <path
                      d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"
                    />
                  </svg>
                </div>
              </div>
            </div>
          </div>
          <div class="flex mt-3">
            <div
              class="bg-white hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 py-1 px-3 rounded shadow cursor-pointer transition-all duration-100"
              @click="update_setting"
            >Confirm</div>
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
              <p class="text-sm text-gray-600 mb-2">
                Given the initial state, the final state, and the description of objects in the initial state.
                The output should be the transformation(s) (
                <code>&lt;object, attribute, value&gt;</code>) that could achieve the shown change.
              </p>
              <p
                class="text-sm text-gray-600 mb-5"
              >Notice: Overlapping and moving out of the invisible area is not allowed throughout the whole transformation process.</p>
            </div>
            <div class="flex items-center">
              <div class="flex flex-col justify-center">
                <div class="border-8 cursor-pointer border-white shadow">
                  <img :src="initial_src" alt class="state" />
                </div>
                <div class="flex flex-row items-center justify-between mt-5">
                  <div class="mx-auto flex">
                    <p class="font-bold">Initial State</p>
                  </div>
                </div>
              </div>
              <div class="flex flex-col justify-center">
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
                <div class="border-8 cursor-pointer border-white shadow">
                  <img :src="final_src" alt class="state" />
                </div>
                <div class="flex flex-col items-center justify-between mt-5">
                  <div class="flex mx-auto">
                    <p class="font-bold">Final State</p>
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
                  v-for="(object, index) in initial_objs"
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
        <div class="flex">
          <div class="max-w-2xl">
            <div class="border-b mb-5">
              <p class="text-lg font-semibold">Your Answer</p>
            </div>
            <p
              class="mb-2 text-sm text-gray-600"
            >Choose correct transformation(s) that can achieve the change. You could add the number of transformation if needed.</p>
            <div class>
              <div class="flex">
                <draggable :list="transformations" handle=".handle">
                  <div class="flex px-3 py-2 my-3 text-blue-800 font-semibold">
                    <span class="cursor-move mx-3" style="min-width:15px">
                      <i class="fa fa-align-justify text-white"></i>
                    </span>
                    <span class="mx-3" style="min-width:45px">Order</span>
                    <span class="mx-3" style="min-width:55px">Object</span>
                    <span class="mx-3" style="min-width:85px">Attribute</span>
                    <span class="mx-3" style="min-width:125px">Value</span>
                  </div>
                  <div
                    v-for="(t, i) in transformations"
                    :key="i"
                    class="flex px-3 py-2 my-3 bg-white rounded shadow"
                  >
                    <span class="handle cursor-move mx-3" style="min-width:15px">
                      <i class="fa fa-align-justify"></i>
                    </span>
                    <span class="mx-3 text-center text-gray-700" style="min-width:45px;">{{ i }}</span>
                    <div class="mx-3" style="min-width:55px;">
                      <select v-model="t.obj_idx">
                        <option v-for="obj in objs" :value="obj" :key="obj">{{ obj }}</option>
                      </select>
                    </div>
                    <div class="mx-3" style="min-width:85px;">
                      <select v-model="t.attr">
                        <option v-for="attr in attrs" :value="attr" :key="attr">{{ attr }}</option>
                      </select>
                    </div>
                    <div class="mx-3" style="min-width:125px;">
                      <select v-model="t.pair">
                        <option
                          v-for="pair in get_pairs(t.attr)"
                          :value="pair"
                          :key="pair"
                        >{{ pair.replace(t.attr+'.', '') }}</option>
                      </select>
                    </div>
                    <span @click="removeAt(i)" class="mx-3 cursor-pointer">
                      <i class="fa fa-times"></i>
                    </span>
                  </div>
                </draggable>

                <div class="ml-5 py-1 flex flex-col justify-end">
                  <div
                    class="bg-white hover:bg-blue-700 border-blue-700 text-blue-700 hover:text-white text-sm border-2 px-2 py-1 my-3 ml-5 rounded-full shadow cursor-pointer transition-all duration-100"
                    @click="add"
                  >
                    <i class="fa fa-plus"></i>
                  </div>
                </div>
              </div>
              <div class="flex mt-5">
                <div
                  class="bg-white hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 py-1 px-3 rounded shadow cursor-pointer transition-all duration-100"
                  @click="submit"
                >Submit</div>
                <div
                  class="ml-5 bg-white hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 py-1 px-3 rounded shadow cursor-pointer transition-all duration-100"
                  @click="next"
                >Next</div>
              </div>
              <div v-show="test_result.target" class="mt-10">
                <div class="border-b mb-5">
                  <p class="text-lg font-semibold">Test Result</p>
                </div>
                <div class="text-gray-600 text-sm">
                  <p>
                    <span class="font-semibold text-blue-800">User:</span>
                    {{ test_result.user }}
                  </p>
                  <p>
                    <span class="font-semibold text-blue-800">Problem:</span>
                    {{ sample.problem }}
                  </p>
                  <div class="flex">
                    <p>
                      <span class="font-semibold text-blue-800">Index:</span>
                      {{ sample.idx }}
                    </p>
                    <p class="ml-5">
                      <router-link
                        class="underline text-blue-800 hover:text-blue-600"
                        target="_blank"
                        :to="{ path: '/dataset', query: {datadir: sample.datadir, index: sample.idx}}"
                      >detail</router-link>
                    </p>
                  </div>
                  <p>
                    <span class="font-semibold text-blue-800">View Final:</span>
                    {{ sample.view_final }}
                  </p>
                  <p>
                    <span class="font-semibold text-blue-800">Time Usage:</span>
                    {{ format_duration(test_result.duration) }}
                  </p>
                  <p class="mt-3">
                    <span class="font-semibold text-blue-800 mr-1">Correctness:</span>
                    <span v-show="test_result.correct">
                      <i class="text-green-500 fas fa-check"></i>
                    </span>
                    <span v-show="!test_result.correct">
                      <i class="text-red-500 fas fa-times"></i>
                    </span>
                  </p>
                  <p
                    v-show="(!test_result.loose_correct) || test_result.err_overlap || test_result.err_invalid_position"
                  >
                    <span class="font-semibold text-blue-800 mr-1">Error Reason:</span>
                    <span
                      v-show="!test_result.loose_correct"
                      class="text-red-500 mx-1"
                    >Incorrect Final State</span>
                    <span
                      v-show="test_result.err_invalid_position"
                      class="text-red-500 mx-1"
                    >Invalid Position</span>
                    <span v-show="test_result.err_overlap" class="text-red-500 mx-1">Overlapping</span>
                  </p>
                  <p class="font-semibold text-blue-800">Ground Truth:</p>
                  <p
                    class="mx-3 px-3 py-1 bg-white rounded m-1 max-w-xs"
                    v-for="(t, i) in (test_result.target || [])"
                    :key="'gt_' + i"
                  >
                    <span class="mx-3 text-blue-700">{{ i }}</span>
                    <span class="mx-3">{{ t[0] }}</span>
                    <span class="mx-3">{{ sample.pairs_list[t[1]] }}</span>
                  </p>
                  <p class="font-semibold text-blue-800">Your Answer:</p>
                  <p
                    class="mx-3 px-3 py-1 bg-white rounded m-1 max-w-xs"
                    v-for="(t, i) in (test_result.pred || [])"
                    :key="'pred_' + i"
                  >
                    <span class="mx-3 text-blue-700">{{ i }}</span>
                    <span class="mx-3">{{ t[0] }}</span>
                    <span class="mx-3">{{ sample.pairs_list[t[1]] }}</span>
                  </p>
                </div>
              </div>
            </div>
          </div>

          <!-- <div class="ml-5" v-show="user == 'test'"> -->
          <div class="ml-5">
            <div class="border-b mb-5">
              <p class="text-lg font-semibold">Assistant Diagram</p>
            </div>
            <p class="text-sm text-gray-600 mb-3">
              <span>
                This diagram can help you understand the rules of this game, such as the coordinate system and the direction.
                It will be hidden for non-
                <code>test</code>
              </span>
              user.
            </p>
            <div id="diagram"></div>
          </div>
        </div>
      </div>
    </div>

    <div>
      <div class="max-w-5xl mx-auto px-5 py-10">
        <div class="border-b mb-5">
          <p class="text-lg font-semibold">Testing History</p>
        </div>
        <p class="my-5 text-gray-600 text-sm">Click the button below to see your testing history.</p>
        <div>
          <div class="flex">
            <div
              class="bg-white hover:bg-blue-700 border-blue-700 text-blue-800 hover:text-white text-sm border-2 py-1 px-3 rounded shadow cursor-pointer transition-all duration-100"
              @click="get_history"
            >Show</div>
          </div>
          <p
            class="mt-5 font-semibold text-blue-800 text-sm"
            v-if="history.samples && history_samples.length==0"
          >No results found.</p>
          <div class="mt-5" v-if="history_samples.length > 0">
            <div class="flex flex-col text-blue-800 text-sm">
              <div class="max-w-lg">
                <div class="flex justify-between">
                  <div class="flex">
                    <p class="font-semibold">Total:</p>
                    <p class="px-2">{{ history.statistics[this.problem].count }}</p>
                  </div>
                  <p v-if="problem == 'view'" class="ml-3">
                    <span>(</span>
                    <span
                      class="px-1"
                      v-for="view in ['Camera_Center', 'Camera_Left', 'Camera_Right']"
                      :key="view + '_count'"
                    >
                      <span class="font-semibold">{{ lodash.split(view, '_')[1][0] }}:</span>
                      <span>{{ history.statistics[view].count }}</span>
                    </span>
                    <span>)</span>
                  </p>
                </div>
                <div class="flex justify-between">
                  <div class="flex">
                    <p class="font-semibold">Average Time:</p>
                    <p class="px-2">{{ format_duration(history.statistics[this.problem].time) }}</p>
                  </div>
                  <p v-if="problem == 'view'" class="ml-3">
                    <span>(</span>
                    <span
                      class="px-1"
                      v-for="view in ['Camera_Center', 'Camera_Left', 'Camera_Right']"
                      :key="view + '_time'"
                    >
                      <span class="font-semibold">{{ lodash.split(view, '_')[1][0] }}:</span>
                      <span>{{ format_duration(history.statistics[view].time) }}</span>
                    </span>
                    <span>)</span>
                  </p>
                </div>
                <div class="flex mt-2 justify-between">
                  <div class="flex">
                    <p class="font-semibold">Acc:</p>
                    <p
                      class="px-2"
                    >{{ lodash.round(history.statistics[this.problem].acc * 100, 2) }}%</p>
                  </div>
                  <p v-if="problem == 'view'" class="ml-3">
                    <span>(</span>
                    <span
                      class="px-1"
                      v-for="view in ['Camera_Center', 'Camera_Left', 'Camera_Right']"
                      :key="view + '_acc'"
                    >
                      <span class="font-semibold">{{ lodash.split(view, '_')[1][0] }}:</span>
                      <span>{{ lodash.round(history.statistics[view].acc * 100, 2) }}%</span>
                    </span>
                    <span>)</span>
                  </p>
                </div>
                <div class="flex items-end justify-between">
                  <div class="flex">
                    <p class="font-semibold">Loose Acc:</p>
                    <p
                      class="px-2"
                    >{{ lodash.round(history.statistics[this.problem].loose_acc * 100, 2) }}%</p>
                    <div class="mb-2 tooltip">
                      <p class="text-gray-500 hover:text-gray-700">
                        <i class="far fa-question-circle"></i>
                      </p>
                      <p
                        class="py-1 px-3 tooltip-text text-gray-600 bg-white rounded border shadow border-blue-700"
                      >Ignore rules like no overlapping and no invalid position.</p>
                    </div>
                  </div>
                  <p v-if="problem == 'view'" class="ml-3">
                    <span>(</span>
                    <span
                      class="px-1"
                      v-for="view in ['Camera_Center', 'Camera_Left', 'Camera_Right']"
                      :key="view + '_loose_acc'"
                    >
                      <span class="font-semibold">{{ lodash.split(view, '_')[1][0] }}:</span>
                      <span>{{ lodash.round(history.statistics[view].loose_acc * 100, 2) }}%</span>
                    </span>
                    <span>)</span>
                  </p>
                </div>
                <template v-if="this.problem == 'basic' || this.problem == 'view'">
                  <div class="flex justify-between">
                    <div class="flex">
                      <p class="font-semibold">ObjAcc:</p>
                      <p
                        class="px-2"
                      >{{ lodash.round(history.statistics[this.problem].obj_acc * 100, 2) }}%</p>
                    </div>
                    <p v-if="problem == 'view'" class="ml-3">
                      <span>(</span>
                      <span
                        class="px-1"
                        v-for="view in ['Camera_Center', 'Camera_Left', 'Camera_Right']"
                        :key="view + '_obj_acc'"
                      >
                        <span class="font-semibold">{{ lodash.split(view, '_')[1][0] }}:</span>
                        <span>{{ lodash.round(history.statistics[view].obj_acc * 100, 2) }}%</span>
                      </span>
                      <span>)</span>
                    </p>
                  </div>
                  <div class="flex justify-between">
                    <div class="flex">
                      <p class="font-semibold">AttrAcc:</p>
                      <p
                        class="px-2"
                      >{{ lodash.round(history.statistics[this.problem].attr_acc * 100, 2) }}%</p>
                    </div>
                    <p v-if="problem == 'view'" class="ml-3">
                      <span>(</span>
                      <span
                        class="px-1"
                        v-for="view in ['Camera_Center', 'Camera_Left', 'Camera_Right']"
                        :key="view + '_attr_acc'"
                      >
                        <span class="font-semibold">{{ lodash.split(view, '_')[1][0] }}:</span>
                        <span>{{ lodash.round(history.statistics[view].attr_acc * 100, 2) }}%</span>
                      </span>
                      <span>)</span>
                    </p>
                  </div>
                  <div class="flex justify-between">
                    <div class="flex">
                      <p class="font-semibold">ValAcc:</p>
                      <p
                        class="px-2"
                      >{{ lodash.round(history.statistics[this.problem].val_acc * 100, 2) }}%</p>
                    </div>
                    <p v-if="problem == 'view'" class="ml-3">
                      <span>(</span>
                      <span
                        class="px-1"
                        v-for="view in ['Camera_Center', 'Camera_Left', 'Camera_Right']"
                        :key="view + '_val_acc'"
                      >
                        <span class="font-semibold">{{ lodash.split(view, '_')[1][0] }}:</span>
                        <span>{{ lodash.round(history.statistics[view].val_acc * 100, 2) }}%</span>
                      </span>
                      <span>)</span>
                    </p>
                  </div>
                </template>
                <template>
                  <div class="flex">
                    <p class="font-semibold">Average Distance:</p>
                    <p class="px-2">{{ lodash.round(history.statistics[this.problem].dist, 2) }}</p>
                  </div>
                  <div class="flex">
                    <p class="font-semibold">Average Normlized Distance:</p>
                    <p
                      class="px-2"
                    >{{ lodash.round(history.statistics[this.problem].norm_dist, 2) }}</p>
                  </div>
                </template>
                <template>
                  <div class="">
                    <p class="font-semibold">Step Accuracy:</p>
                    <div>
                      <div v-for="(acc, step) in history.statistics[this.problem].step" :key="step" class="flex">
                        <p class="px-2">{{ step }}:</p>
                        <p class="px-2">{{ lodash.round(acc * 100, 2) }}%</p>
                      </div>
                    </div>
                  </div>
                </template>
              </div>
            </div>
            <table id="history" class="text-sm mt-5">
              <thead class="text-blue-800">
                <tr>
                  <th class="pb-3 px-3"></th>
                  <th class="pb-3 px-3">Date</th>
                  <th class="pb-3 px-3">Index</th>
                  <th class="pb-3 px-3">View Final</th>
                  <th class="pb-3 px-3">Time Cost</th>
                  <th class="pb-3 px-3">Prediction</th>
                  <th class="pb-3 px-3">Grount Truth</th>
                  <th class="pb-3 px-3">Error Reason</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="(s, i) in history_samples">
                  <tr
                    class="items-center shadow rounded history text-right"
                    v-bind:class="{
                    'bg-green-100': s.result.correct,
                    'bg-red-100': !s.result.correct
                  }"
                    :key="s.sample.idx"
                  >
                    <td class="font-semibold">{{i}}</td>
                    <td class="text-gray-600">{{ format_time(s.end_timestamp) }}</td>
                    <td>
                      <router-link
                        class="underline text-blue-800 hover:text-blue-600"
                        target="_blank"
                        :to="{ path: '/dataset', query: {datadir: s.sample.datadir, index: s.sample.idx}}"
                      >{{ s.sample.idx }}</router-link>
                    </td>
                    <td class>{{ s.sample.view_final }}</td>
                    <td>{{ format_duration(s.duration) }}</td>
                    <td class="text-left text-xs">
                      <p v-for="(t, i) in (s.result.pred || [])" :key="s.sample.idx + '_pred_' + i">
                        <span class="mx-1 text-gray-700">{{ i }}</span>
                        <span class="mx-1">{{ t[0] }}</span>
                        <span class="mx-1">{{ s.sample.pairs_list[t[1]] }}</span>
                      </p>
                    </td>
                    <td class="text-left text-xs">
                      <p v-for="(t, i) in (s.result.target || [])" :key="s.sample.idx + '_gt_' + i">
                        <span class="mx-1 text-gray-700">{{ i }}</span>
                        <span class="mx-1">{{ t[0] }}</span>
                        <span class="mx-1">{{ s.sample.pairs_list[t[1]] }}</span>
                      </p>
                    </td>
                    <td class="text-left">
                      <p v-show="!s.result.loose_correct" class="text-red-500">Incorrect Final State</p>
                      <p
                        v-show="s.result.err_invalid_position"
                        class="text-red-500"
                      >Invalid Position</p>
                      <p v-show="s.result.err_overlap" class="text-red-500">Overlapping</p>
                    </td>
                  </tr>
                  <tr :key="'blank_' + i">
                    <td class="py-1 pb-2"></td>
                  </tr>
                </template>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import _ from 'lodash';
import draggable from 'vuedraggable';

import { API } from '../js/api';
import { Diagram } from '../js/diagram';

export default {
  data() {
    return {
      problems: [],
      input_user: 'test',
      input_problem: '',
      user: 'test',
      problem: '',
      datadir: '',
      loading_info: false,
      info: {},
      sample: {},
      diagram_scale: 4,
      lodash: _,
      transformations: [],
      default_transformation: { obj_idx: 0, attr: 'position', pair: 'position.front.1' },
      test_result: {},
      history: {},
    };
  },
  components: {
    draggable,
  },
  mounted: function() {
    var self = this;
    axios.get(API.api_problems()).then(function(response) {
      self.problems = response.data.problems;
      if (self.problems.length > 0) {
        self.input_problem = self.problem = self.problems[0];
        self.get_sample();
      }
    });
  },
  methods: {
    update_setting: function() {
      this.user = this.input_user;
      if (this.problem != this.input_problem) {
        this.history = {};
      }
      this.problem = this.input_problem;
      this.get_sample();
    },
    get_sample: function() {
      if (!this.loading_info) {
        var self = this;
        this.loading_info = true;
        axios
          .get(API.api_random_info(this.problem, this.user))
          .then(function(response) {
            if (!_.isEqual(self.sample, response.data)) {
              self.sample = response.data;
              self.info = response.data.info;
              self.transformations = [_.clone(self.default_transformation)];
              self.test_result = {};
              self.sample_start = _.now();
            }
          })
          .finally(() => (this.loading_info = false));
      }
    },
    get_pairs: function(attr) {
      if (
        typeof this.sample.pairs != 'undefined' &&
        typeof attr != 'undefined'
      ) {
        return this.sample.pairs[attr];
      } else {
        return [];
      }
    },
    add: function() {
      this.transformations.push(_.clone(this.default_transformation));
    },
    removeAt: function(idx) {
      this.transformations.splice(idx, 1);
    },
    submit: function() {
      let self = this;
      var time_now = _.now();
      axios
        .post(
          API.api_submit_test(),
          {
            transformations: self.transformations,
            sample: self.sample,
            end_timestamp: time_now,
            end_time: new Date(time_now).toString(),
            duration: time_now - self.sample_start,
          },
          {
            headers: {
              'Content-Type': 'application/json;charset=UTF-8',
              'Access-Control-Allow-Origin': '*',
            },
          }
        )
        .then(function(response) {
          self.test_result = response.data;
        });
    },
    next: function() {
      this.get_sample();
    },
    get_history: function() {
      let self = this;
      axios
        .get(API.api_test_history(this.user, this.problem))
        .then(function(response) {
          self.history = response.data;
        });
    },
    format_time: function(timestamp) {
      var t = new Date(timestamp);
      var time_str =
        `${t.getFullYear()}-${t.getMonth()}-${t.getDay()} ` +
        `${t.getHours()}:${t.getMinutes()}:${t.getSeconds()}`;
      return time_str;
    },
    format_duration: function(duration) {
      var gap = [1000, 60, 60];
      var unit = ['ms', 's', 'min', 'h'];
      var time_str = '';

      for (var i in gap) {
        time_str = (duration % gap[i]) + ' ' + unit[i] + ' ' + time_str;
        duration = parseInt(duration / gap[i]);
        if (duration == 0) {
          break;
        } else if (i == 0) {
          time_str = '';
        }
      }

      return time_str;
    },
  },
  watch: {
    info: function() {
      // if (this.user == 'test') {
        Diagram.update_diagram(
          'diagram',
          this.initial_objs,
          this.diagram_scale
        );
    },
  },
  computed: {
    initial_src: function() {
      if (typeof this.info.states != 'undefined') {
        return API.api_image(this.sample.datadir, this.info.states[0].image);
      } else {
        return '';
      }
    },
    final_src: function() {
      if (typeof this.info.states != 'undefined') {
        return API.api_image(this.sample.datadir, this.info.states[1].image);
      } else {
        return '';
      }
    },
    initial_objs: function() {
      if (typeof this.info.states != 'undefined') {
        return this.info.states[0].objects;
      } else {
        return [];
      }
    },
    objs: function() {
      if (typeof this.sample.objs != 'undefined') {
        return this.sample.objs;
      } else {
        return [];
      }
    },
    attrs: function() {
      if (typeof this.sample.pairs != 'undefined') {
        return _.keys(this.sample.pairs);
      } else {
        return [];
      }
    },
    history_samples: function() {
      if (this.history.samples) {
        return this.history.samples[this.problem];
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

<style scoped>
.history td {
  padding: 0.35em 1em;
}

.tooltip .tooltip-text {
  visibility: hidden;
  text-align: center;
  position: absolute;
  z-index: 100;
}
.tooltip:hover .tooltip-text {
  visibility: visible;
}
</style>

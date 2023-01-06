<template>
  <div>
    <div class="py-16">
      <div class="mx-auto max-w-5xl px-5 flex justify-between">
        <div>
          <p class="text-5xl font-semibold">TVR</p>
          <p class="text-xl font-sans text-blue-900">
            Transformation Driven Visual Reasoning
          </p>
          <div class="flex mt-2">
            <div
              class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800
           hover:text-white text-sm border-2 py-1 px-3 rounded shadow
          cursor-pointer transition-all duration-100 mr-2"
            >
              <a href="">Download TRANCE</a>
            </div>
            <div
              class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800
           hover:text-white text-sm border-2 py-1 px-3 rounded shadow
          cursor-pointer transition-all duration-100 mr-2"
            >
              TRANCE Settings
            </div>
            <div
              class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800
           hover:text-white text-sm border-2 py-1 px-3 rounded shadow
          cursor-pointer transition-all duration-100 mr-2"
            >
              Explore Dataset
            </div>
            <div
              class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800
           hover:text-white text-sm border-2 py-1 px-3 rounded shadow
          cursor-pointer transition-all duration-100 mr-2"
            >
              BibTex
            </div>
            <div
              class="flex-shrink-0 hover:bg-blue-700 border-blue-700 text-blue-800
           hover:text-white text-sm border-2 py-1 px-3 rounded shadow
          cursor-pointer transition-all duration-100 mr-2"
            >
              Paper
            </div>
          </div>
        </div>
        <div class="w-64 grid grid-cols-3 gap-4 h-20">
          <div class="flex flex-col justify-center">
            <a href="http://www.bigdatalab.ac.cn/lab/">
              <img src="@/assets/image/NDST_logo.png" alt="" />
            </a>
          </div>
          <div class="flex flex-col justify-center">
            <a href="http://www.ict.ac.cn/">
              <img src="@/assets/image/ICT_logo.png" alt="" />
            </a>
          </div>
          <div class="flex flex-col justify-center">
            <a href="http://english.ucas.ac.cn/">
              <img src="@/assets/image/CAS_logo.png" alt="" />
            </a>
          </div>
        </div>
      </div>
    </div>

    <div class="bg-gray-100 shadow-inner">
      <div class="mx-auto max-w-5xl px-5 py-16">
        <div class="flex justify-between">
          <div class=" max-w-md">
            <p class="mb-5 font-semibold text-xl text-center">
              Task Introduction
            </p>
            <p class="text-justify">
              Piaget’s cognitive theory tells us human intelligence must have
              functions to represent both the transformational and static
              aspects of reality. Transformation driven visual reasoning is
              designed to emphasize the importance of transformation in visual
              reasoing. We create TRANCE dataset with three different settings
              to study this problem. Given both the initial and ﬁnal states, the
              target is to infer the corresponding single-step or multi-step
              transformation, represented as a triplet (object, attribute,
              value) or a sequence of triplets, respectively.
            </p>
          </div>
          <div>
            <div
              class="grid grid-cols-3 justify-around text-center bg-white
        border-gray-200  shadow-inner
        rounded-full items-center cursor-default"
            >
              <div
                class="border-r p-5 border-gray-200 bg-red-300 rounded-full
          transition-all duration-100 shadow-md hover:shadow-2xl col-span-1"
              >
                Initial State
              </div>
              <div
                class="col-span-1 text-4xl px-5 self-center font-serif font-bold"
              >
                ?
              </div>
              <div
                class="border-l p-5 border-gray-200 bg-blue-300 rounded-full
                transition-all duration-100 shadow-md hover:shadow-2xl col-span-1"
              >
                Final State
              </div>
            </div>
            <p class="text-lg text-center mt-5">
              Which<br />
              transformation(s)<br />
              could change <br />
              the initial state to the final state
            </p>
          </div>
        </div>
      </div>
    </div>

    <div class="text-center">
      <div class="mx-auto max-w-5xl px-5 py-16">
        <p class="mb-5 font-semibold text-xl">Three settings of TRANCE</p>

        <div class="flex py-10 justify-between">
          <div
            class="grid grid-cols-2 gap-2 items-center text-center max-w-md
           rounded-lg text-lg"
          >
            <div class="">
              <img
                class="shadow-lg"
                src="@/assets/image/basic_move_initial.png"
                alt=""
              />
              <p class="mt-2">Initial State</p>
            </div>
            <div class="">
              <img
                class="shadow-lg"
                src="@/assets/image/basic_move_initial.png"
                alt=""
              />
              <p class="mt-2">Final State</p>
            </div>
          </div>
          <div class=" max-w-md">
            <p class="mb-5 font-semibold text-xl text-center">Basic</p>
            <p class="text-justify">
              Given a pair of images, which are constructed from changing one of
              the ﬁve attributes of an object in the initial image. Our task is
              to ﬁnd out which attribute of which object has been changed to
              which value, then generate a transformation triplet (object,
              attribute, value).
            </p>
          </div>
        </div>

        <div class="flex py-10 justify-between flex-row-reverse">
          <div
            class="grid grid-cols-2 gap-2 items-center text-center max-w-md
          rounded-lg text-lg"
          >
            <div class="">
              <img
                class="shadow-lg"
                src="@/assets/image/basic_move_initial.png"
                alt=""
              />
              <p class="mt-2">Initial State</p>
            </div>
            <div class="">
              <img
                class="shadow-lg"
                src="@/assets/image/basic_move_initial.png"
                alt=""
              />
              <p class="mt-2">Final State</p>
            </div>
          </div>
          <div class=" max-w-md">
            <p class="mb-5 font-semibold text-xl text-center">View</p>
            <p class="text-justify">
              The setting of Basic is 2D, which is too simple to detect
              human-like real world transformations, such as 3D spatial changes.
              To tackle this problem, we extend the Basic setting to View we set
              three ﬁxed cameras, placed in the left side, center, and right
              side of the plane. The initial state is always captured by the
              center camera. While for the ﬁnal state, we use all three cameras.
              Thus, we have three ﬁnal states for each initial state.
            </p>
          </div>
        </div>

        <div class="flex pt-10 justify-between">
          <div
            class="grid grid-cols-2 gap-2 items-center text-center max-w-md
          rounded-lg text-lg"
          >
            <div class="">
              <img
                class="shadow-lg"
                src="@/assets/image/basic_move_initial.png"
                alt=""
              />
              <p class="mt-2">Initial State</p>
            </div>
            <div class="">
              <img
                class="shadow-lg"
                src="@/assets/image/basic_move_initial.png"
                alt=""
              />
              <p class="mt-2">Final State</p>
            </div>
          </div>
          <div class=" max-w-md">
            <p class="mb-5 font-semibold text-xl text-center">Event</p>
            <p class="text-justify">
              In the real world, it is very common that multiple transformations
              exist between two states. Therefore we construct this multi-step
              transformation setting to test whether machines can handle this
              situation. The goal of a vision system is to predict a sequence of
              transformations that could reproduce the same ﬁnal state from the
              initial state.
            </p>
          </div>
        </div>
      </div>
    </div>

    <div class="bg-gray-100 shadow-inner">
      <div class="mx-auto max-w-5xl px-5 py-16">
        <div class="flex justify-between">
          <div class="">
            <p class="mb-5 font-semibold text-xl text-center">BibTex</p>
            <pre class="bg-white p-5 rounded border border-blue-700 shadow">
inproceedings{hong2020tvr,
    title={Transformation Driven Visual Reasoning},
    author={Xin Hong, Yanyan Lan, Liang Pang, Jiafeng Guo and Xueqi Cheng},
    year = {2020},
    journal={}
}</pre
            >
          </div>
          <div class="">
            <p class="mb-5 font-semibold text-xl text-center">PDF</p>
            <div
              class="bg-white border border-blue-700 h-40 w-32
          text-center rounded"
            ></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script></script>

<style></style>

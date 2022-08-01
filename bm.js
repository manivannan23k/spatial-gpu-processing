const benchmark = require('@gpujs/benchmark');
const { GPU } = require('gpu.js');

const benchmarks = benchmark.benchmark({
    cpu: new GPU({mode: 'cpu'}),
    gpu: new GPU()
});

console.log((benchmarks))
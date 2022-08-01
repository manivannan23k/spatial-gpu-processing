import { GPU, CPUKernel } from 'gpu.js';
import RasterUtils from './raster-utils.js';
import * as fs from 'fs';
import { createCanvas } from 'canvas';
import proj4 from 'proj4';


const gpu = new GPU({
    mode: 'gpu'
});

let COLOR_RAMP = [
    [200,200,200,255],
    [0,0,0,255]
];

function Math_nearestNeighbour(values, y, x, width, height) {
    y = Math.round(y);
    x = Math.round(x);
    if (y < 0 || x < 0 || y > height || x > width) {
        return 0;
    }
    y = Math.max(Math.floor(y), 0);
    x = Math.max(Math.floor(x), 0);
    y = Math.min(Math.ceil(y), height - 1);
    x = Math.min(Math.ceil(x), width - 1);
    return values[y][x];
}

function Math_bicubicInterpolation(values, x, y, width, height) {
    function bicubicInterpolation(r1, r2, r3, r4, dx, dy){
        function cubicInterpolation(p, dx){
            return p[1] + 0.5 * dx*(p[2] - p[0] + dx*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + dx*(3.0*(p[1] - p[2]) + p[3] - p[0])));
        }
        return cubicInterpolation([
            cubicInterpolation(r1, dx), 
            cubicInterpolation(r2, dx), 
            cubicInterpolation(r3, dx), 
            cubicInterpolation(r4, dx)
        ], dy);
    }
    const m = height;
    const n = width;
    if(x<=1 || x>=n-2 || y<=1 || y>=m-2){
        return 0;
    }
    let _x = Math.floor(x);
    let _y = Math.floor(y);
    let r1 = [
        values[_y-1][_x-1], values[_y-1][_x], values[_y-1][_x+1], values[_y-1][_x+2]
    ];
    let r2 = [
        values[_y][_x-1], values[_y][_x], values[_y][_x+1], values[_y][_x+2]
    ];
    let r3 = [
        values[_y+1][_x-1], values[_y+1][_x], values[_y+1][_x+1], values[_y+1][_x+2]
    ];
    let r4 = [
        values[_y+2][_x-1], values[_y+2][_x], values[_y+2][_x+1], values[_y+2][_x+2]
    ];
    let dx = x - _x, dy = y - _y;

    return bicubicInterpolation(
        r1, r2, r3, r4, dx, dy
    );

}

function Math_bilinearInterpolation(values, x, y, width, height) {
    let x1 = Math.floor(x - 1), y1 = Math.floor(y - 1), x2 = Math.ceil(x + 1), y2 = Math.ceil(y + 1);
    x1 = Math.max(0, x1);
    x2 = Math.max(0, x2);
    y1 = Math.max(0, y1);
    y2 = Math.max(0, y2);
    x1 = Math.min(height - 1, x1);
    x2 = Math.min(height - 1, x2);
    y1 = Math.min(width - 1, y1);
    y2 = Math.min(width - 1, y2);
    let q11 = (((x2 - x) * (y2 - y)) / ((x2 - x1) * (y2 - y1))) * values[x1][y1]
    let q21 = (((x - x1) * (y2 - y)) / ((x2 - x1) * (y2 - y1))) * values[x2][y1]
    let q12 = (((x2 - x) * (y - y1)) / ((x2 - x1) * (y2 - y1))) * values[x1][y2]
    let q22 = (((x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))) * values[x2][y2]
    return q11 + q21 + q12 + q22
}

function getColorAtRatio(endColor, startColor, localRatio){
    const w = localRatio * 2 - 1;
    const w1 = (w + 1) / 2;
    const w2 = 1 - w1;
    return [Math.round(endColor[0] * w1 + startColor[0] * w2),
        Math.round(endColor[1] * w1 + startColor[1] * w2),
        Math.round(endColor[2] * w1 + startColor[2] * w2),
        Math.round((endColor[3] * w1 + startColor[3] * w2))];
}

gpu.addFunction(Math_nearestNeighbour)
gpu.addFunction(Math_bicubicInterpolation)
gpu.addFunction(Math_bilinearInterpolation)
gpu.addFunction(getColorAtRatio)

const renderKernel = gpu.createKernel(function(dataArr, min, max) {
    let overAllRatio = (dataArr[this.thread.y][this.thread.x] - min) / (max - min);
    const colorRampSize = 2;
    const rampRatio = (1 / (colorRampSize - 1));
    const index = Math.floor(overAllRatio / rampRatio);
    const startColorIndex = index;
    var endColorIndex = startColorIndex + 1;
    if(endColorIndex>=colorRampSize){
        endColorIndex = startColorIndex;
    }
    const startColorX = index * rampRatio;
    const endColorX = (index + 1) * rampRatio;
    const localRatio = (overAllRatio - startColorX) / (endColorX - startColorX);


    let color = getColorAtRatio(
        [
            this.constants.colorRamp[startColorIndex][0],
            this.constants.colorRamp[startColorIndex][1],
            this.constants.colorRamp[startColorIndex][2],
            this.constants.colorRamp[startColorIndex][3]
        ]
        ,
        [

            this.constants.colorRamp[endColorIndex][0],
            this.constants.colorRamp[endColorIndex][1],
            this.constants.colorRamp[endColorIndex][2],
            this.constants.colorRamp[endColorIndex][3]
        ] 
        , 
        localRatio);
    this.color(
        color[0]/255,
        color[1]/255,
        color[2]/255,
        color[3]/255
    )
})
.setConstants({
    colorRamp: COLOR_RAMP
});
const renderCpu = (dataArr, width, height, min, max, colorRamp) => {
    const buffer = new Uint8ClampedArray(width * height * 4);

    for(let y=0;y<height;y++){
        for(let x=0;x<width;x++){
            let overAllRatio = (dataArr[y][x] - min) / (max - min);
            if(overAllRatio<0) overAllRatio=0
            if(overAllRatio>1) overAllRatio=1
            const colorRampSize = colorRamp.length;
            const rampRatio = (1 / (colorRampSize - 1));
            const index = Math.floor(overAllRatio / rampRatio);
            const startColorIndex = index;
            var endColorIndex = startColorIndex + 1;
            if(endColorIndex>=colorRampSize){
                endColorIndex = startColorIndex;
            }
            const startColorX = index * rampRatio;
            const endColorX = (index + 1) * rampRatio;
            const localRatio = (overAllRatio - startColorX) / (endColorX - startColorX);
            // console.log(dataArr[y][x])
            let v = getColorAtRatio(
                [
                    colorRamp[startColorIndex][0],
                    colorRamp[startColorIndex][1],
                    colorRamp[startColorIndex][2],
                    colorRamp[startColorIndex][3]
                ]
                ,
                [

                    colorRamp[endColorIndex][0],
                    colorRamp[endColorIndex][1],
                    colorRamp[endColorIndex][2],
                    colorRamp[endColorIndex][3]
                ] 
                , 
                localRatio);
            let pos = ((y * width) + x) * 4;
            buffer[pos] = v[0];
            buffer[pos + 1] = v[1];
            buffer[pos + 2] = v[2];
            buffer[pos + 3] = v[3];
        }
    }
    return buffer;
}

const ndviKernel = gpu.createKernel(function(b4DataArr, b5DataArr, outNx, outNy, nx, ny) {

    let ix = (nx * this.thread.x/outNx), iy = (ny * this.thread.y/outNy);

    const b4v = Math_bicubicInterpolation(b4DataArr, iy, ix, nx, ny);
    const b5v = Math_bicubicInterpolation(b5DataArr, iy, ix, nx, ny);
    if(b5v===b4v){
        return -1;
    }
    const ndvi = (b5v-b4v)/(b5v+b4v);
    return ndvi;
});

const ndviCpu = (b4DataArr, b5DataArr, outNx, outNy, nx, ny) => {
    let outputArr = [];
    for(let y=0;y<outNy;y++){
        outputArr[y] = [];
        for(let x=0;x<outNx;x++){
            let ix = (nx * x/outNx), iy = (ny * y/outNy);
            const b4v = Math_bicubicInterpolation(b4DataArr, iy, ix, nx, ny);
            const b5v = Math_bicubicInterpolation(b5DataArr, iy, ix, nx, ny);
            let v = (b5v-b4v)/(b5v+b4v);
            if(isNaN(v)){
                v = -1;
            }
            outputArr[y][x] = v;
        }
    }
    return outputArr;
}

const start = async () => {
    let b4Data = await RasterUtils.processTiffSource('../data/band4_2.tif');
    let b5Data = await RasterUtils.processTiffSource('../data/band5_2.tif');


    let outData = {
        xRes: b4Data.xRes,
        yRes: b4Data.yRes,
        x1: b4Data.x1,
        y1: b4Data.y1,
        x2: b4Data.x2,
        y2: b4Data.y2,
    };
    outData['nx'] = Math.floor((outData.x2 - outData.x1)/outData.xRes);
    outData['ny'] = Math.floor((outData.y2 - outData.y1)/outData.yRes);
    
    console.log(`${outData.nx} x ${outData.ny} (${outData.nx * outData.ny})`);
    let result = null;
    console.time('GPU')
    result = ndviKernel
        .setOutput([
            outData.nx, outData.ny
        ])(
            b4Data.rasterData,
            b5Data.rasterData,
            outData.nx,
            outData.ny,
            b4Data.nx,
            b4Data.ny
        );
    console.timeEnd('GPU');

    outData.min = NaN;
    outData.max = NaN;
    for(let y=0;y<outData.ny;y++){
        for(let x=0;x<outData.nx;x++){
            if(isNaN(outData.min) || outData.min > result[y][x]){
                outData.min = result[y][x];
            }
            if(isNaN(outData.max) || outData.max < result[y][x]){
                outData.max = result[y][x];
            }
        }
    }
    // console.log(outData.min, outData.max)

    console.time('CPU')
    result = ndviCpu(
        b4Data.rasterData,
        b5Data.rasterData,
        outData.nx,
        outData.ny,
        b4Data.nx,
        b4Data.ny
    );
    console.timeEnd('CPU')

    // console.log(result)

    // let buffer = null;
    // let canvas = createCanvas(outData.nx, outData.ny);

    // renderKernel
    // .setOutput([outData.nx, outData.ny])
    // .setGraphical(true);
    // console.time("GPU Render")
    // renderKernel(result, outData.min, outData.max);
    // buffer = renderKernel.getPixels();
    // console.timeEnd("GPU Render")


    // console.time("CPU Render")
    // buffer = renderCpu(result, outData.nx, outData.ny, outData.min, outData.max, COLOR_RAMP);
    // console.timeEnd("CPU Render")

    // const ctx = canvas.getContext('2d');
    // const iData = ctx.createImageData(outData.nx, outData.ny);
    // iData.data.set(buffer);
    // ctx.imageSmoothingEnabled = false;
    // ctx.putImageData(iData, 0, 0);

    // const dataUrl = canvas.toDataURL();
    // let base64Data = dataUrl.replace(/^data:image\/png;base64,/, "");
    // fs.writeFileSync('../data/out.png', base64Data, {encoding: 'base64'});

}

start()
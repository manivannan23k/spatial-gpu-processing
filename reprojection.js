import { GPU, CPUKernel } from 'gpu.js';
import RasterUtils from './raster-utils.js';
import * as fs from 'fs';
import { createCanvas } from 'canvas';
import proj4 from 'proj4';


const gpu = new GPU({
    mode: 'gpu'
});

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

gpu.addFunction(Math_nearestNeighbour)
gpu.addFunction(Math_bicubicInterpolation)
gpu.addFunction(Math_bilinearInterpolation)

const reprojectKernel = gpu.createKernel(function(
    dataArr,
    outNx,
    outNy,
    outX1,
    outY1,
    outXRes,
    outYRes,
    nx,
    ny,
    x1,
    y1,
    xRes,
    yRes,
    fromSrs,
    toSrs
    ) {

    function _4326to3857(x, y){
        const ox = (x * 20037508.34 / 180)
        const oy = (Math.log(Math.tan((90 + y) * Math.PI / 360)) / (Math.PI / 180)) * (20037508.34 / 180)
        return [ox, oy]
    }

    function _3857to4326(x, y){
        const ox = x * 180 / 20037508.34;
        const oy = 360/Math.PI * Math.atan(Math.exp(y / (20037508.34 / 180)  * (Math.PI/180))) - 90;
        return [ox, oy]
    }
    var ix = nx * this.thread.x/outNx, iy = ny * this.thread.y/outNy;
    if(fromSrs!==toSrs){
        let lat = x1, lon = y1, tempCoords = [0,0],
            _x = outX1 + (outXRes * this.thread.x), _y = outY1 + (outYRes * this.thread.y),
            inX = _x, inY = _y;
        if(toSrs===3857){
            tempCoords = _3857to4326(_x, _y);
            lon = tempCoords[0];
            lat = tempCoords[1];
        }else{
            lat = _y;
            lon = _x;
        }
        if(fromSrs===3857){
            tempCoords = _4326to3857(lon, lat);
            inX = tempCoords[0];
            inY = tempCoords[1];
        }else{
            inX = lon;
            inY = lat;
        }
        ix = Math.floor((inX - x1)/xRes);
        iy = Math.floor((inY - y1)/yRes);
    }

    return Math_bilinearInterpolation(dataArr, iy, ix, nx, ny);
});

const reprojectCpu = (
    dataArr,
    outNx,
    outNy,
    outX1,
    outY1,
    outXRes,
    outYRes,
    nx,
    ny,
    x1,
    y1,
    xRes,
    yRes,
    fromSrs,
    toSrs
) => {
    function _4326to3857(x, y){
        const ox = (x * 20037508.34 / 180)
        const oy = (Math.log(Math.tan((90 + y) * Math.PI / 360)) / (Math.PI / 180)) * (20037508.34 / 180)
        return [ox, oy]
    }

    function _3857to4326(x, y){
        const ox = x * 180 / 20037508.34;
        const oy = 360/Math.PI * Math.atan(Math.exp(y / (20037508.34 / 180)  * (Math.PI/180))) - 90;
        return [ox, oy]
    }

    let outputArr = [];
    for(let y=0;y<outNy;y++){
        outputArr[y] = [];
        for(let x=0;x<outNx;x++){

            var ix = nx * x/outNx, iy = ny * y/outNy;
            if(fromSrs!==toSrs){
                let lat = x1, lon = y1, tempCoords = [0,0],
                    _x = outX1 + (outXRes * x), _y = outY1 + (outYRes * y),
                    inX = _x, inY = _y;
                if(toSrs===3857){
                    tempCoords = _3857to4326(_x, _y);
                    lon = tempCoords[0];
                    lat = tempCoords[1];
                }else{
                    lat = _y;
                    lon = _x;
                }
                if(fromSrs===3857){
                    tempCoords = _4326to3857(lon, lat);
                    inX = tempCoords[0];
                    inY = tempCoords[1];
                }else{
                    inX = lon;
                    inY = lat;
                }
                ix = Math.floor((inX - x1)/xRes);
                iy = Math.floor((inY - y1)/yRes);
            }
        
            let v = Math_bilinearInterpolation(dataArr, iy, ix, nx, ny);

            // let ix = nx * x/outNx, iy = ny * y/outNy;
            // let v = Math_bicubicInterpolation(dataArr, iy, ix, nx, ny);


            outputArr[y][x] = v;


        }
    }
    return outputArr;
}

const start = async (outSrs) => {
    let data = await RasterUtils.processTiffSource('../data/dem3.tif');

    let prj = proj4(`EPSG:${data.coordinateSystem}`, `EPSG:${3857}`);
    let nw = prj.forward([data.x1, data.y1]),
        se = prj.forward([data.x2, data.y2]);
    let outBbox = [
        nw[0], nw[1], se[0], se[1]
    ];

    let outNx, outNy, outXRes, outYRes;
    let diagCount = Math.floor(Math.sqrt(
        Math.pow(data.nx, 2) + Math.pow(data.ny, 2)
    ));

    let outRes = Math.sqrt(
        Math.pow(outBbox[0] - outBbox[2], 2) + Math.pow(outBbox[1] - outBbox[3], 2)
    )/diagCount;
    outXRes = outRes;
    outYRes = -outRes;
    outNx = Math.ceil(
        (outBbox[2] - outBbox[0])/outXRes
    );
    outNy = Math.ceil(
        (outBbox[3] - outBbox[1])/outYRes
    );

    // console.log(outNx, outNy, outBbox)

    let outData = {
        xRes: outXRes,
        yRes: outYRes,
        x1: outBbox[0],
        y1: outBbox[1],
        x2: outBbox[2],
        y2: outBbox[3],
        nx: outNx,
        ny: outNy
    };

    console.log(`${outData.nx} X ${outData.ny} pixels: ${outData.nx * outData.ny}`)
    console.time('GPU')
    let result = reprojectKernel
        .setOutput([
            outData.nx, outData.ny
        ])(
            data.rasterData,
            outData.nx,
            outData.ny,
            outData.x1,
            outData.y1,
            outData.xRes,
            outData.yRes,
            data.nx,
            data.ny,
            data.x1,
            data.y1,
            data.xRes,
            data.yRes,
            data.coordinateSystem,
            outSrs
        );
    console.timeEnd('GPU')
    // console.log(result)


    console.time('CPU')
    result = reprojectCpu(
        data.rasterData,
            outData.nx,
            outData.ny,
            outData.x1,
            outData.y1,
            outData.xRes,
            outData.yRes,
            data.nx,
            data.ny,
            data.x1,
            data.y1,
            data.xRes,
            data.yRes,
            data.coordinateSystem,
            outSrs
    );
    console.timeEnd('CPU')

    // console.log(result)


}

start(3857)
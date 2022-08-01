import {fromArrayBuffer} from 'geotiff';
// import { writeGeotiff } from './geotiff/writer';
// import proj4 from './proj4-full';
import * as fs from 'fs';

const COLOUR_RAMPS = {
    "bw": [
        "#000000FF",
        "#ffffffFF",
    ],
    "rainbow": [
        "#0000ffFF",
        "#007dffFF",
        "#00ffffFF",
        "#00ff7dFF",
        "#00ff00FF",
        "#7dff00FF",
        "#ffff00FF",
        "#ff7d00FF",
        "#ff0000FF"
    ],
    "windyTemp":[
        "#A7E6FFFF",
        "#5ECDF4FF",
        "#49AA29FF",
        "#C4E013FF",
        "#F0CF18FF",
        "#FAAF19FF",
        "#F88D33FF",
        "#CC460BFF",
    ],
    "water": [
        "#E2F5FFFF",
        "#56C5FFFF",
        "#56C5FFFF"
    ],
    "bwr": [
        "#457DC9FF",
        "#FFFFFFFF",
        "#C95C45FF"
    ],
    "ocean": [
        "#000050ff",
        "#001e64ff",
        "#003266ff",
        "#0052a5ff",
        "#87cefaff"
    ],
    "terrain": [
        "#006147ff",
        "#e8d67dff",
        "#821e1eff",
        "#cececeff",
        "#ffffffff"
    ]
};

const ALLOWED_PROJECTIONS = [
    4326,
    3857
]

const parseGeoTiffData = async (ab) => {
    const tiff = await fromArrayBuffer(ab);
    const image = await tiff.getImage();
    const bbox = image.getBoundingBox();
    const x1 = bbox[0],
        y1 = bbox[3],
        x2 = bbox[2],
        y2 = bbox[1],
        xRes = image.getResolution()[0],
        yRes = image.getResolution()[1],
        nx = image.getWidth(),
        ny = image.getHeight();

    let data = {
        x1, y1, x2, y2, xRes, yRes, nx, ny,
        min: null,
        max: null,
        coordinateSystem: image.geoKeys?(image.geoKeys["GeographicTypeGeoKey"]  || image.geoKeys["ProjectedCSTypeGeoKey"]):null,
        geographicTypeGeoKey: image.geoKeys?image.geoKeys["GeographicTypeGeoKey"]:null,
        projectedCSTypeGeoKey: image.geoKeys?image.geoKeys["ProjectedCSTypeGeoKey"]:null,
        rasterData: [],
        bitsPerSample: image.getBitsPerSample(),
        sampleFormat: image.getSampleFormat()
    };

    let bands = await image.readRasters();

    const nb = bands.length;

    if (nb===0){
        return
    }

    for (let bandNo = 0; bandNo < bands.length; bandNo++) {
        const band = bands[bandNo];
        data.rasterData = [];
        for (let y = 0; y < ny; y++) {
            data.rasterData[y] = [];
            for (let x = 0; x < nx; x++) {
                const v = band[(y*bands.width)+x];
                if(isNaN(v)){
                    v = -1;
                }
                if (data.min===null || data.min > v)
                    data.min = v;
                if (data.max===null || data.max < v)
                    data.max = v;
                data.rasterData[y][x] = v;
            }
        }
    }
    return data;
}

const urlToTiffData = (url) => {
    return new Promise((resolve, reject) => {
        fetch(url)
        .then(r=>r.arrayBuffer())
        .then(async (ab)=>{
            let data = await parseGeoTiffData(ab);
            return resolve(data);
        })
        .catch(e=>{
            console.log(e);
            return reject(e);
        });
    })
}

const pathToTiffData = async (url) => {
    let data = fs.readFileSync(url);
    data = await parseGeoTiffData(bufferToArrayBuffer(data));
    return data;
}

const bufferToArrayBuffer = (buf) => {
    const ab = new ArrayBuffer(buf.length);
    const view = new Uint8Array(ab);
    for (let i = 0; i < buf.length; ++i) {
        view[i] = buf[i];
    }
    return ab;
}

const validUrl = (str) => {
    const pattern = new RegExp("^((https|http|ftp|rtsp|mms)?://)"
        + "?(([0-9a-z_!~*'().&=+$%-]+: )?[0-9a-z_!~*'().&=+$%-]+@)?"
        + "(([0-9]{1,3}\.){3}[0-9]{1,3}"
        + "|"
        + "([0-9a-z_!~*'()-]+\.)*"
        + "([0-9a-z][0-9a-z-]{0,61})?[0-9a-z]\."
        + "[a-z]{2,6})"
        + "(:[0-9]{1,4})?"
        + "((/?)|"
        + "(/[0-9a-z_!~*'().;?:@&=+$,%#-]+)+/?)$");
    return !!pattern.test(str);
}

const validPath = (string) => {
    return fs.existsSync(string);
}

const processTiffSource = async (data) => {
    switch (data.constructor.name) {
        case "String":
            if (validUrl(data)) {
                return await urlToTiffData(data);
            }else if (validPath(data)) {
                return await pathToTiffData(data);
            }
            throw "Not a valid url";
        case "ArrayBuffer":
            return await parseGeoTiffData(data);
        case "Object":
            return (data);
    }
    throw "Not a valid data";
}


// const getValueAt = (x, y, layer, method) => {
//     let xi =  (x - layer.x1)/layer.xRes, yi =  (y - layer.y1)/layer.yRes;
//     if(!method){
//         method = "nearestNeighbour";
//     }
//     switch(method){
//         case "nearestNeighbour":
//             return Math.nearestNeighbour(layer.rasterData, yi, xi) || 0;
//         case "bilinear":
//             return Math.bilinearInterpolation(layer.rasterData, yi, xi) || 0;
//         case "bicubic":
//             try{
//                 return Math.bicubicInterpolation(layer.rasterData, yi, xi) || 0;
//             }catch(e){
//                 return Math.nearestNeighbour(layer.rasterData, yi, xi) || 0;
//             }
//     }
    
// }

// const getDefaultWarp = (data, targetSrs, options) => {
//     if(!options){
//         options = {};
//     }
//     let prj = proj4(`EPSG:${data.coordinateSystem}`, `EPSG:${targetSrs}`);
//     let nw = prj.forward([data.x1, data.y1]),
//         se = prj.forward([data.x2, data.y2]);

//     let outNx, outNy, outBbox, outXRes, outYRes;

    
    
//     if(options.x1 && options.y1 && options.x2 && options.y2){
//         outBbox = [
//             options.x1, options.y1, options.x2, options.y2
//         ];
//     }else{
//         outBbox = [
//             nw[0], nw[1], se[0], se[1]
//         ];
//     }

//     if(options.xRes && options.yRes){
//         outXRes = options.xRes;
//         outYRes = options.yRes;

//         outNx = Math.ceil(
//             (outBbox[2] - outBbox[0])/outXRes
//         );
//         outNy = Math.ceil(
//             (outBbox[3] - outBbox[1])/outYRes
//         );
//     }else if(options.nx && options.ny){
//         outNx = options.nx;
//         outNy = options.ny;
        
//         outXRes = (
//             (outBbox[2] - outBbox[0])/outNx
//         );
//         outYRes = (
//             (outBbox[3] - outBbox[1])/outNy
//         );
//     }else{
//         let diagCount = Math.floor(Math.sqrt(
//             Math.pow(data.nx, 2) + Math.pow(data.ny, 2)
//         ));
        
//         let outRes = Math.sqrt(
//             Math.pow(outBbox[0] - outBbox[2], 2) + Math.pow(outBbox[1] - outBbox[3], 2)
//         )/diagCount;
//         outXRes = outRes;
//         outYRes = -outRes;
//         outNx = Math.ceil(
//             (outBbox[2] - outBbox[0])/outXRes
//         );
//         outNy = Math.ceil(
//             (outBbox[3] - outBbox[1])/outYRes
//         );
//     }

//     return {
//         coordinateSystem: targetSrs,
//         max: null,
//         min: null,
//         nx: outNx,
//         ny: outNy,
//         rasterData: [],
//         x1: outBbox[0],
//         y1: outBbox[1],
//         x2: outBbox[2],
//         y2: outBbox[3],
//         xRes: outXRes,
//         yRes: outYRes,
//         bitsPerSample: data.bitsPerSample,
//         sampleFormat: data.sampleFormat,
//         geographicTypeGeoKey: data.geographicTypeGeoKey,
//         projectedCSTypeGeoKey: data.projectedCSTypeGeoKey
//     }
// }

// const reproject = async (source, targetSrs, options) => {
//     try{

//         if(!options){
//             options = {};
//         }
//         if(ALLOWED_PROJECTIONS.indexOf(targetSrs) === -1){
//             return {
//                 error: true,
//                 message: "TARGET_PROJECTION_NOT_SUPPORTED",
//                 data: null
//             }
//         }

//         const data = await processTiffSource(source);

//         if(ALLOWED_PROJECTIONS.indexOf(data.coordinateSystem) === -1){
//             return {
//                 error: true,
//                 message: "SOURCE_PROJECTION_NOT_SUPPORTED",
//                 data: null
//             }
//         }

//         let prj = proj4(`EPSG:${data.coordinateSystem}`, `EPSG:${targetSrs}`);

//         let _data = getDefaultWarp(data, targetSrs, options)

//         for(let y=0;y<_data.ny;y++){
//             _data.rasterData[y] = [];
//             for(let x=0;x<_data.nx;x++){
//                 let _ox = _data.x1 + x * _data.xRes, _oy = _data.y1 + y * _data.yRes;
//                 let _ic = prj.inverse([_ox, _oy]);
//                 let v = getValueAt(_ic[0], _ic[1], data, options.interpolation);
//                 _data.rasterData[y][x] = v;
//                 if (_data.min===null || _data.min > v)
//                     _data.min = v;
//                 if (_data.max===null || _data.max < v)
//                     _data.max = v;
//             }
//         }
//         return {
//             error: false,
//             message: "SUCCESS",
//             data: _data
//         };
//     }catch(e){
//         console.log(e);
//         return {
//             error: true,
//             message: "EXCEPTION",
//             data: null
//         };
//     }
// }

// const rasterObjToAb = (data) => {
//     let crsKeys = {};
//     switch(data.coordinateSystem){
//         case 4326:
//             crsKeys['GeographicTypeGeoKey'] = data.coordinateSystem;
//             break;
//         case 3857:
//             crsKeys['ProjectedCSTypeGeoKey'] = data.coordinateSystem;
//             break;
//     }
//     let metadata = {
//         width: data.nx,
//         height: data.ny,
//         BitsPerSample: [data.bitsPerSample],
//         SampleFormat: [data.sampleFormat],
//         ModelPixelScale: [data.xRes, -data.yRes, 0],
//         ModelTiepoint: [0, 0, 0, data.x1, data.y1, 0],
//         ...crsKeys
//     };
//     let rData = [data.rasterData];
//     let ab = writeGeotiff(rData, metadata);
//     return ab;
// }

// const saveRasterObjAsTiff = (data, filename) => {
//     let ab = rasterObjToAb(data);
//     saveAbAsTiff(filename, ab);
// }

// function saveAbAsTiff(fname, byte) {
//     var blob = new Blob([byte], {type: "image/tiff"});
//     var link = document.createElement('a');
//     link.href = window.URL.createObjectURL(blob);
//     link.download = fname;
//     link.click();
//     // document.removeChild(link);
// };

const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    let rgba = result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
        a: parseInt(result[4], 16)
    } : null;
    if (!rgba)
        return rgba;
    return [rgba.r, rgba.g, rgba.b, rgba.a]
}

export default {
    // reproject,
    // getDefaultWarp,
    // rasterObjToAb,
    // saveAbAsTiff,
    // saveRasterObjAsTiff,
    processTiffSource,
    hexToRgb,
    COLOUR_RAMPS
}
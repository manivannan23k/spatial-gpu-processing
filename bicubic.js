const cubicInterpolation = (p, dx) => {
    return p[1] + 0.5 * dx*(p[2] - p[0] + dx*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + dx*(3.0*(p[1] - p[2]) + p[3] - p[0])));
}

const bicubicInterpolation = (r1, r2, r3, r4, dx, dy) => {
    let v1 = cubicInterpolation(r1, dx);
    let v2 = cubicInterpolation(r2, dx);
    let v3 = cubicInterpolation(r3, dx);
    let v4 = cubicInterpolation(r4, dx);
    return cubicInterpolation([v1, v2, v3, v4], dy);
}

const calcBicubic = (values, x, y, width, height) => {
    let blX = Math.floor(x);
    let blY = Math.floor(y);
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
    let v = bicubicInterpolation(
        r1, r2, r3, r4, dx, dy
    )

    return v;

}

const data = [
    [1,2,3,4,1,2,3,4],
    [5,6,7,8,5,6,7,8],
    [9,10,11,12,9,10,11,12],
    [13,14,15,16,13,14,15,16],
    [1,2,3,4,1,2,3,4],
    [5,6,7,8,5,6,7,8],
    [9,10,11,12,9,10,11,12],
    [13,14,15,16,13,14,15,16]
];
const nx = 10, ny = 10, inNx = 8, inNy = 8;

for(let y=0;y<ny;y++){
    for(let x=0;x<nx;x++){
        let xi = inNx * x/nx;
        let yi = inNy * y/ny;
        let v = calcBicubic(data, xi, yi, inNx, inNy);
        console.log(v);
    }
}

// const dx = -1, dy = -1;


// let x = bicubicInterpolation(data, dx, dy);
// console.log(x);
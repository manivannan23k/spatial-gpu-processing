const MathInterpolation = () => {
    if (!Math.bilinearInterpolation) {
        Math.bilinearInterpolation = function (values, x, y) {
            if (x < 0 || y < 0 || x > values.length || y > values[0].length) {
                // console.log(0, x, y)
                return null;
            }
            let x1 = Math.floor(x - 1), y1 = Math.floor(y - 1), x2 = Math.ceil(x + 1), y2 = Math.ceil(y + 1);
            x1 = Math.max(0, x1);
            x2 = Math.max(0, x2);
            y1 = Math.max(0, y1);
            y2 = Math.max(0, y2);
            x1 = Math.min(values.length - 1, x1);
            x2 = Math.min(values.length - 1, x2);
            y1 = Math.min(values[0].length - 1, y1);
            y2 = Math.min(values[0].length - 1, y2);
            // try{
            let q11 = (((x2 - x) * (y2 - y)) / ((x2 - x1) * (y2 - y1))) * values[x1][y1]
            let q21 = (((x - x1) * (y2 - y)) / ((x2 - x1) * (y2 - y1))) * values[x2][y1]
            let q12 = (((x2 - x) * (y - y1)) / ((x2 - x1) * (y2 - y1))) * values[x1][y2]
            let q22 = (((x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))) * values[x2][y2]
            return q11 + q21 + q12 + q22
            // }catch(e){
            //     console.log(x1, x2, y1, y2);
            // }
        }
    }
    if (!Math.nearestNeighbour) {
        Math.nearestNeighbour = function (values, x, y) {
            x = Math.round(x);
            y = Math.round(y);
            if (x < 0 || y < 0 || x > values.length || y > values[0].length) {
                return 0;
            }
            x = Math.max(Math.floor(x), 0);
            y = Math.max(Math.floor(y), 0);
            x = Math.min(Math.ceil(x), values.length - 1);
            y = Math.min(Math.ceil(y), values[0].length - 1);
            return values[x][y];
        }
    }
    if (!Math.bicubicInterpolation) {
        Math.bicubicInterpolation = function (values, dx, dy) {

            const createInterpolator = (values, options = {}) => {
                options = Object.assign({
                    extrapolate: false,
                    scaleX: 1,
                    scaleY: 1,
                    translateX: 0,
                    translateY: 0
                }, options);
                const a00 = values[1][1],
                    a01 = (-1 / 2) * values[1][0] + (1 / 2) * values[1][2],
                    a02 = values[1][0] + (-5 / 2) * values[1][1] + 2 * values[1][2] + (-1 / 2) * values[1][3],
                    a03 = (-1 / 2) * values[1][0] + (3 / 2) * values[1][1] + (-3 / 2) * values[1][2] + (1 / 2) * values[1][3],
                    a10 = (-1 / 2) * values[0][1] + (1 / 2) * values[2][1],
                    a11 = (1 / 4) * values[0][0] + (-1 / 4) * values[0][2] + (-1 / 4) * values[2][0] + (1 / 4) * values[2][2],
                    a12 = (-1 / 2) * values[0][0] + (5 / 4) * values[0][1] + (-1) * values[0][2] + (1 / 4) * values[0][3] + (1 / 2) * values[2][0] + (-5 / 4) * values[2][1] + values[2][2] + (-1 / 4) * values[2][3],
                    a13 = (1 / 4) * values[0][0] + (-3 / 4) * values[0][1] + (3 / 4) * values[0][2] + (-1 / 4) * values[0][3] + (-1 / 4) * values[2][0] + (3 / 4) * values[2][1] + (-3 / 4) * values[2][2] + (1 / 4) * values[2][3],
                    a20 = values[0][1] + (-5 / 2) * values[1][1] + 2 * values[2][1] + (-1 / 2) * values[3][1],
                    a21 = (-1 / 2) * values[0][0] + (1 / 2) * values[0][2] + (5 / 4) * values[1][0] + (-5 / 4) * values[1][2] + (-1) * values[2][0] + values[2][2] + (1 / 4) * values[3][0] + (-1 / 4) * values[3][2],
                    a22 = values[0][0] + (-5 / 2) * values[0][1] + 2 * values[0][2] + (-1 / 2) * values[0][3] + (-5 / 2) * values[1][0] + (25 / 4) * values[1][1] + (-5) * values[1][2] + (5 / 4) * values[1][3] + 2 * values[2][0] + (-5) * values[2][1] + 4 * values[2][2] + (-1) * values[2][3] + (-1 / 2) * values[3][0] + (5 / 4) * values[3][1] + (-1) * values[3][2] + (1 / 4) * values[3][3],
                    a23 = (-1 / 2) * values[0][0] + (3 / 2) * values[0][1] + (-3 / 2) * values[0][2] + (1 / 2) * values[0][3] + (5 / 4) * values[1][0] + (-15 / 4) * values[1][1] + (15 / 4) * values[1][2] + (-5 / 4) * values[1][3] + (-1) * values[2][0] + 3 * values[2][1] + (-3) * values[2][2] + values[2][3] + (1 / 4) * values[3][0] + (-3 / 4) * values[3][1] + (3 / 4) * values[3][2] + (-1 / 4) * values[3][3],
                    a30 = (-1 / 2) * values[0][1] + (3 / 2) * values[1][1] + (-3 / 2) * values[2][1] + (1 / 2) * values[3][1],
                    a31 = (1 / 4) * values[0][0] + (-1 / 4) * values[0][2] + (-3 / 4) * values[1][0] + (3 / 4) * values[1][2] + (3 / 4) * values[2][0] + (-3 / 4) * values[2][2] + (-1 / 4) * values[3][0] + (1 / 4) * values[3][2],
                    a32 = (-1 / 2) * values[0][0] + (5 / 4) * values[0][1] + (-1) * values[0][2] + (1 / 4) * values[0][3] + (3 / 2) * values[1][0] + (-15 / 4) * values[1][1] + 3 * values[1][2] + (-3 / 4) * values[1][3] + (-3 / 2) * values[2][0] + (15 / 4) * values[2][1] + (-3) * values[2][2] + (3 / 4) * values[2][3] + (1 / 2) * values[3][0] + (-5 / 4) * values[3][1] + values[3][2] + (-1 / 4) * values[3][3],
                    a33 = (1 / 4) * values[0][0] + (-3 / 4) * values[0][1] + (3 / 4) * values[0][2] + (-1 / 4) * values[0][3] + (-3 / 4) * values[1][0] + (9 / 4) * values[1][1] + (-9 / 4) * values[1][2] + (3 / 4) * values[1][3] + (3 / 4) * values[2][0] + (-9 / 4) * values[2][1] + (9 / 4) * values[2][2] + (-3 / 4) * values[2][3] + (-1 / 4) * values[3][0] + (3 / 4) * values[3][1] + (-3 / 4) * values[3][2] + (1 / 4) * values[3][3];

                return (x, y) => {
                    x = (x * options.scaleX) + options.translateX;
                    y = (y * options.scaleY) + options.translateY;

                    if (x < 0 || y < 0 || x > 1 || y > 1) throw 'cannot interpolate outside the square from (0, 0) to (1, 1): (' + x + ', ' + y + ')';

                    const x2 = x * x,
                        x3 = x * x2,
                        y2 = y * y,
                        y3 = y * y2;

                    return (a00 + a01 * y + a02 * y2 + a03 * y3) +
                        (a10 + a11 * y + a12 * y2 + a13 * y3) * x +
                        (a20 + a21 * y + a22 * y2 + a23 * y3) * x2 +
                        (a30 + a31 * y + a32 * y2 + a33 * y3) * x3;
                }
            }

            const createGridInterpolator = (values, options = {}) => {
                let x;
                options = Object.assign({
                    extrapolate: false,
                    scaleX: 1,
                    scaleY: 1,
                    translateX: 0,
                    translateY: 0
                }, options);

                const m = values.length;
                const n = values[0].length;
                const interpolators = [];

                if (options.extrapolate) {
                    //Extrapolate X
                    values[-2] = [];
                    values[-1] = [];
                    values[m] = [];
                    values[m + 1] = [];
                    for (let y = 0; y < n; y++) {
                        const leftDelta = values[0][y] - values[1][y];
                        const rightDelta = values[m - 1][y] - values[m - 2][y];
                        values[-2][y] = values[0][y] + 2 * leftDelta;
                        values[-1][y] = values[0][y] + leftDelta;
                        values[m][y] = values[m - 1][y] + rightDelta;
                        values[m + 1][y] = values[m - 1][y] + 2 * rightDelta;
                    }

                    //Extrapolate Y
                    for (x = -2; x < m + 2; x++) {
                        const bottomDelta = values[x][0] - values[x][1];
                        const topDelta = values[x][n - 1] - values[x][n - 2];
                        values[x][-2] = values[x][0] + 2 * bottomDelta;
                        values[x][-1] = values[x][0] + bottomDelta;
                        values[x][n] = values[x][n - 1] + topDelta;
                        values[x][n + 1] = values[x][n - 1] + 2 * topDelta;
                    }

                    //Populate interpolator arrays
                    for (x = -1; x < m; x++) interpolators[x] = [];
                } else {
                    //Populate interpolator arrays
                    for (x = 1; x < m - 2; x++) interpolators[x] = [];
                }

                return (x, y) => {
                    x = (x * options.scaleX) + options.translateX;
                    y = (y * options.scaleY) + options.translateY;

                    if (options.extrapolate) {
                        if (x < -1 || y < -1 || x > m || y > n) throw 'cannot interpolate outside the rectangle from (-1, -1) to (' + m + ', ' + n + ') even when extrapolating: (' + x + ', ' + y + ')';
                    } else {
                        if (x < 1 || y < 1 || x > m - 2 || y > n - 2) throw 'cannot interpolate outside the rectangle from (1, 1) to (' + (m - 2) + ', ' + (n - 2) + '): (' + x + ', ' + y + '), you might want to enable extrapolating';
                    }

                    let blX = Math.floor(x);// The position of interpolator's (0, 0) for this point
                    let blY = Math.floor(y);

                    if (options.extrapolate) {//If you're trying to interpolate on the top or right edges of what can be interpolated, you have to interpolate in the region to the left or bottom respectively.
                        if (x === m) blX--;
                        if (y === n) blY--;
                    } else {
                        if (x === m - 2) blX--;
                        if (y === n - 2) blY--;
                    }


                    if (!interpolators[blX][blY]) {
                        interpolators[blX][blY] = createInterpolator([
                            [values[blX - 1][blY - 1], values[blX - 1][blY], values[blX - 1][blY + 1], values[blX - 1][blY + 2]],
                            [values[blX][blY - 1], values[blX][blY], values[blX][blY + 1], values[blX][blY + 2]],
                            [values[blX + 1][blY - 1], values[blX + 1][blY], values[blX + 1][blY + 1], values[blX + 1][blY + 2]],
                            [values[blX + 2][blY - 1], values[blX + 2][blY], values[blX + 2][blY + 1], values[blX + 2][blY + 2]]
                        ], {
                            translateX: -blX,
                            translateY: -blY
                        });
                    }
                    const interpolator = interpolators[blX][blY];

                    return interpolator(x, y);
                }
            }

            return createGridInterpolator(values)(dx, dy);
        }
    }
}

export default MathInterpolation;
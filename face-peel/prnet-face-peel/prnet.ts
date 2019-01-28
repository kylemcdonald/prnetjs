import * as tf from '@tensorflow/tfjs'

const scale = 255

const pad = 'same';

function bn(input, vars, name) {
    return tf.batchNormalization(
        input,
        vars[name + '/BatchNorm/moving_mean'],
        vars[name + '/BatchNorm/moving_variance'],
        0.001,
        vars[name + '/BatchNorm/gamma'],
        vars[name + '/BatchNorm/beta']
    );
}

// hack to fix https://github.com/tensorflow/tfjs/issues/261#issuecomment-390463379
function conv2d(input, weights, stride, pad) {
    // return tf.conv2d(input, weights, stride, pad);
    var output = tf.conv2d(input, weights, 1, pad);
    if (stride == 1) {
        return output;
    }
    return tf.stridedSlice(output, [0, 0, 0, 0], output.shape, [1, stride, stride, 1]);
}

// https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d
function resBlock(name, vars, x, num_outputs, stride) {
    var shortcut = x;
    if (stride != 1 || x.shape[3] != num_outputs) {
        shortcut = conv2d(shortcut, vars[name + '/shortcut/weights'], stride, pad);
    }
    return tf.tidy(() => {
        const rb_c0 = tf.tidy(() => tf.relu(bn(
            tf.conv2d(x, vars[name + '/Conv/weights'], 1, pad),
            vars, name + '/Conv')));
        const rb_c1 = tf.tidy(() => tf.relu(bn(
            conv2d(rb_c0, vars[name + '/Conv_1/weights'], stride, pad),
            vars, name + '/Conv_1'))) as tf.Tensor4D;
        const rb_c2 = tf.tidy(() => tf.conv2d(rb_c1, vars[name + '/Conv_2/weights'], 1, pad));
        const output = tf.relu(bn(
            shortcut.add(rb_c2),
            vars, name));

        // check(name, output);

        return output as tf.Tensor4D;
    })
}

// based on https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d_transpose
function conv2dTranspose(name, vars, x: tf.Tensor4D, num_outputs: number, stride: number, activation_fn?) {
    if (typeof (activation_fn) === 'undefined') {
        activation_fn = tf.relu;
    }

    const output_shape = [
        x.shape[0],
        x.shape[1] * stride,
        x.shape[2] * stride,
        num_outputs
    ] as [number, number, number, number];

    var output = tf.tidy(() => activation_fn(bn(
        tf.conv2dTranspose(x, vars[name + '/weights'], output_shape, stride, pad),
        vars, name)));

    // check(name, output);

    return output;
}

async function loadWeightMap(dir: string) {
    const weights_manifest_url = dir + '/weights_manifest-simple.json';
    const manifest = await fetch(weights_manifest_url);
    const weightManifest = await manifest.json();
    const weightMap = await tf.io.loadWeights(weightManifest, dir);
    return weightMap;
}


export class PRNet {

    mask: tf.Tensor
    vars: tf.NamedTensorMap
    constructor(public weightsURL: string, public maskImg: HTMLImageElement) { }

    public async setup() {
        this.vars = await this.loadModel()
        this.mask = this.loadMask()
    }

    public forward(x: tf.Tensor4D) {
        return tf.tidy(() => {
            const vars = this.vars;
            const size = 16;
            x = tf.conv2d(x, vars['Conv/weights'] as tf.Tensor4D, 1, pad);
            // check('initial conv', x)
            x = tf.relu(bn(x, vars, 'Conv')) as tf.Tensor4D
            // check('initial bn + relu', x)
            x = resBlock('resBlock', vars, x, size * 2, 2)
            x = resBlock('resBlock_1', vars, x, size * 2, 1) // 128 x 128 x 32
            x = resBlock('resBlock_2', vars, x, size * 4, 2) // 64 x 64 x 64
            x = resBlock('resBlock_3', vars, x, size * 4, 1) // 64 x 64 x 64
            x = resBlock('resBlock_4', vars, x, size * 8, 2) // 32 x 32 x 128
            x = resBlock('resBlock_5', vars, x, size * 8, 1) // 32 x 32 x 128
            x = resBlock('resBlock_6', vars, x, size * 16, 2) // 16 x 16 x 256
            x = resBlock('resBlock_7', vars, x, size * 16, 1) // 16 x 16 x 256
            x = resBlock('resBlock_8', vars, x, size * 32, 2) // 8 x 8 x 512
            x = resBlock('resBlock_9', vars, x, size * 32, 1) // 8 x 8 x 512
        
            x = conv2dTranspose('Conv2d_transpose', vars, x, size * 32, 1) // 8 x 8 x 512 
            x = conv2dTranspose('Conv2d_transpose_1', vars, x, size * 16, 2) // 16 x 16 x 256 
            x = conv2dTranspose('Conv2d_transpose_2', vars, x, size * 16, 1) // 16 x 16 x 256 
            x = conv2dTranspose('Conv2d_transpose_3', vars, x, size * 16, 1) // 16 x 16 x 256 
            x = conv2dTranspose('Conv2d_transpose_4', vars, x, size * 8, 2) // 32 x 32 x 128 
            x = conv2dTranspose('Conv2d_transpose_5', vars, x, size * 8, 1) // 32 x 32 x 128 
            x = conv2dTranspose('Conv2d_transpose_6', vars, x, size * 8, 1) // 32 x 32 x 128 
            x = conv2dTranspose('Conv2d_transpose_7', vars, x, size * 4, 2) // 64 x 64 x 64 
            x = conv2dTranspose('Conv2d_transpose_8', vars, x, size * 4, 1) // 64 x 64 x 64 
            x = conv2dTranspose('Conv2d_transpose_9', vars, x, size * 4, 1) // 64 x 64 x 64 
            x = conv2dTranspose('Conv2d_transpose_10', vars, x, size * 2, 2) // 128 x 128 x 32
            x = conv2dTranspose('Conv2d_transpose_11', vars, x, size * 2, 1) // 128 x 128 x 32
            x = conv2dTranspose('Conv2d_transpose_12', vars, x, size, 2) // 256 x 256 x 16
            x = conv2dTranspose('Conv2d_transpose_13', vars, x, size, 1) // 256 x 256 x 16
            x = conv2dTranspose('Conv2d_transpose_14', vars, x, 3, 1) // 256 x 256 x 3
            x = conv2dTranspose('Conv2d_transpose_15', vars, x, 3, 1) // 256 x 256 x 3
            x = conv2dTranspose('Conv2d_transpose_16', vars, x, 3, 1, tf.sigmoid) // 256 x 256 x 3
        
            x = tf.tidy(() => x.mul(scale).mul(tf.scalar(1.1))) // 1.1 is essential
            // x = x.add(tf.scalar(10)); // ? closer, but unclear why this is needed
            //x = x.transpose([0, 2, 1, 3])
            // x =
            x = tf.tidy(() => x.mul(this.mask))
            return x
        })
    }

    public dispose() {
        tf.dispose([this.vars, this.mask])
    }

    async loadModel() {
        return await loadWeightMap(this.weightsURL)
    }

    loadMask() {
        return tf.tidy(() => {
            let m = tf.fromPixels(this.maskImg)
            m = m.mean(2).expandDims(2) // convert to grayscale
            m = m.asType('float32').div(scale) // convert to float
            return m;
        })
    }
}
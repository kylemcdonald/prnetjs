import * as tf from '@tensorflow/tfjs'
import { PRNet } from './prnet'
import { Tensor } from '@tensorflow/tfjs';

function log(msg) {
    console.log(msg);
    document.getElementById('log').innerHTML += msg + '<br/>';
}

log('starting');

async function setupWebcam(videoElt: HTMLVideoElement) {
    if (navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        videoElt.srcObject = stream
        videoElt.width = 320
        videoElt.height = 240
        return videoElt
    } else {
        log('Could not setup webcam')
        throw new Error('Could not setup webcam')
    }
}

function squareCrop<T extends Tensor>(img: T): T {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

async function main() {
    const outputCanvas = document.querySelector('#output') as HTMLCanvasElement
    const cam = await setupWebcam(document.querySelector('#webcam'))
    const maskImg = document.querySelector('#maskImg') as HTMLImageElement
    const connie = document.querySelector('#connieConverse') as HTMLImageElement
    const net = new PRNet('//localhost:8181/output-0.6.5-quantized', maskImg)
    
    await net.setup()

    while (true) {
        console.time('loop')
    
        const input = tf.tidy(() => {
            let img = squareCrop(tf.fromPixels(cam))
            img = tf.image.resizeBilinear(img, [256, 256])
            img = img.toFloat().div(255)
            return img
        })
    
        const result = tf.tidy(() => net.forward(input.expandDims(0)).squeeze()) as tf.Tensor3D
    
        const gathered = tf.tidy(() => {
            let r = tf.tidy(() => result.toInt().clipByValue(0, 256))
            const [x, y, z] = tf.unstack(r, 2)
            const idx = tf.stack([y, x], 2)
            const gathered = tf.gatherND(input, idx)
            return gathered
        }) as tf.Tensor3D


        const resultVis = tf.tidy(() => result.div(255).clipByValue(0, 1)) as tf.Tensor3D
    
        await tf.toPixels(gathered, outputCanvas)

        console.timeEnd('loop')

        tf.dispose([input, result, gathered, resultVis])
    }
}

main()
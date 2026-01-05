/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.examples.detection.MainActivity;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;


/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 * <p>
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class YoloV5Classifier implements Classifier {

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param isQuantized   Boolean representing model is quantized or not
     */
    public static YoloV5Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final boolean isQuantized,
            final int inputSize
            /*final int[] output_width,
            final int[][] masks,
            final int[] anchors*/)
            throws IOException {
        final YoloV5Classifier d = new YoloV5Classifier();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();

        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            if (isNNAPI) {
                d.nnapiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    d.nnapiDelegate = new NnApiDelegate();
                    options.addDelegate(d.nnapiDelegate);
                    options.setNumThreads(NUM_THREADS);
                    options.setUseNNAPI(true);
                }
            }
            if (isGPU) {
                try (CompatibilityList compatList = new CompatibilityList()) {
                    if(compatList.isDelegateSupportedOnThisDevice()){
                        // if the device has a supported GPU, add the GPU delegate
                        GpuDelegate gpuDelegate = new GpuDelegate();
                        options.addDelegate(gpuDelegate);
                    } else {
                        // if the GPU is not supported, run on 4 threads
                        options.setNumThreads(4);
                    }
                }
            }
            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            d.tfLite = new Interpreter(d.tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.INPUT_SIZE = inputSize;
        d.imgData = ByteBuffer.allocateDirect(d.INPUT_SIZE * d.INPUT_SIZE * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.INPUT_SIZE * d.INPUT_SIZE];

        d.output_box = (int) ((Math.pow((inputSize / 32.0), 2) + Math.pow((inputSize / 16.0), 2) + Math.pow((inputSize / 8.0), 2)) * 3);
        if (d.isModelQuantized){
            Tensor inpten = d.tfLite.getInputTensor(0);
            d.inp_scale = inpten.quantizationParams().getScale();
            d.inp_zero_point = inpten.quantizationParams().getZeroPoint();
            Tensor oupten = d.tfLite.getOutputTensor(0);
            d.oup_scale = oupten.quantizationParams().getScale();
            d.oup_zero_point = oupten.quantizationParams().getZeroPoint();
        }

        int[] shape = d.tfLite.getOutputTensor(0).shape();
        int numClass = shape[shape.length - 1] - 5;
        d.numClass = numClass;
        d.outData = ByteBuffer.allocateDirect(d.output_box * (numClass + 5) * numBytesPerChannel);
        d.outData.order(ByteOrder.nativeOrder());
        return d;
    }

    public int getInputSize() {
        return INPUT_SIZE;
    }
    @Override
    public void enableStatLogging(final boolean logStats) {
    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
        tfLite.close();
        tfLite = null;
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnapiDelegate != null) {
            nnapiDelegate.close();
            nnapiDelegate = null;
        }
        tfliteModel = null;
    }

    public void setNumThreads(int num_threads) {
        if (tfLite != null) {
            tfliteOptions.setNumThreads(num_threads);
            recreateInterpreter();
        }
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) {
            tfliteOptions.setUseNNAPI(isChecked);
            recreateInterpreter();
        }
    }

    private void recreateInterpreter() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = new GpuDelegate();
            tfliteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
        }
    }

    public void useCPU() {
        recreateInterpreter();
    }

    public void useNNAPI() {
        nnapiDelegate = new NnApiDelegate();
        tfliteOptions.addDelegate(nnapiDelegate);
        recreateInterpreter();
    }

    @Override
    public float getObjThresh() {
        return 0.7f;
    }

    private static final Logger LOGGER = new Logger();

    // Float model
    private final float IMAGE_MEAN = 0;

    private final float IMAGE_STD = 255.0f;

    //config yolo
    private int INPUT_SIZE = -1;

    private  int output_box;

    // Number of threads in the java app
    private static final int NUM_THREADS = 1;
    private static boolean isNNAPI = false;
    private static boolean isGPU = false;

    private boolean isModelQuantized;

    /** holds a gpu delegate */
    GpuDelegate gpuDelegate = null;
    /** holds an nnapi delegate */
    NnApiDelegate nnapiDelegate = null;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    // Config values.

    // Pre-allocated buffers.
    private final Vector<String> labels = new Vector<>();
    private int[] intValues;

    private ByteBuffer imgData;
    private ByteBuffer outData;

    private Interpreter tfLite;
    private float inp_scale;
    private int inp_zero_point;
    private float oup_scale;
    private int oup_zero_point;
    private int numClass;
    private YoloV5Classifier() {
    }

    //non maximum suppression
    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<>(
                            50,
                            (Comparator<Recognition>) (lhs, rhs) -> {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (!pq.isEmpty()) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float mNmsThresh = 0.6f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = Math.max(l1, l2);
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = Math.min(r1, r2);
        return right - left;
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) ((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) (((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        return imgData;
    }

    public ArrayList<Recognition> recognizeImage(Bitmap bitmap) {
        convertBitmapToByteBuffer(bitmap);

        Map<Integer, Object> outputMap = new HashMap<>();

        outData.rewind();
        outputMap.put(0, outData);
        Log.d("YoloV5Classifier", "mObjThresh: " + getObjThresh());

        Object[] inputArray = {imgData};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
        byteBuffer.rewind();

        ArrayList<Recognition> detections = new ArrayList<>();

        float[][][] out = new float[1][output_box][numClass + 5];
        Log.d("YoloV5Classifier", "out[0] detect start");
        for (int i = 0; i < output_box; ++i) {
            for (int j = 0; j < numClass + 5; ++j) {
                if (isModelQuantized){
                    out[0][i][j] = oup_scale * (((int) byteBuffer.get() & 0xFF) - oup_zero_point);
                }
                else {
                    out[0][i][j] = byteBuffer.getFloat();
                }
            }
            // Denormalize xywh
            for (int j = 0; j < 4; ++j) {
                out[0][i][j] *= getInputSize();
            }
        }
        for (int i = 0; i < output_box; ++i){
            final int offset = 0;
            final float confidence = out[0][i][4];
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];
            System.arraycopy(out[0][i], 5, classes, 0, labels.size());

            for (int c = 0; c < labels.size(); ++c) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }

            final float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > getObjThresh()) {
                final float xPos = out[0][i][0];
                final float yPos = out[0][i][1];

                final float w = out[0][i][2];
                final float h = out[0][i][3];
                Log.d("YoloV5Classifier",
                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        confidenceInClass, rect, detectedClass));
            }
        }

        Log.d("YoloV5Classifier", "detect end");
        return nms(detections);
    }

    public float[] restore_box(float[] pred_coor, float oriW, float oriH, int intputSize) {
        // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        float resize_ratioW = (float) (1.0 * intputSize / oriW);
        float resize_ratioH = (float) (1.0 * intputSize / oriH);

        float resize_ratio = Math.min(resize_ratioW, resize_ratioH);

        float dw = (intputSize - resize_ratio * oriW) / 2;
        float dh = (intputSize - resize_ratio * oriH) / 2;

        pred_coor[0] = (pred_coor[0] - dw) / resize_ratio;
        pred_coor[2] = (pred_coor[2] - dw) / resize_ratio;

        pred_coor[1] = (pred_coor[1] - dh) / resize_ratio;
        pred_coor[3] = (pred_coor[3] - dh) / resize_ratio;

        // (3) clip some boxes those are out of range
        pred_coor[0] = Math.max(0, pred_coor[0]);
        pred_coor[1] = Math.max(0, pred_coor[1]);

        pred_coor[2] = Math.min(oriW - 1, pred_coor[2]);
        pred_coor[3] = Math.min(oriH - 1, pred_coor[3]);

        return pred_coor;
    }
}

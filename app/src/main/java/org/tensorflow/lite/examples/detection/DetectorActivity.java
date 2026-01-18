package org.tensorflow.lite.examples.detection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

public class DetectorActivity extends CameraActivity implements ImageReader.OnImageAvailableListener {
    private static final String TAG = "DetectorActivity";

    private Classifier detector;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private boolean computingDetection = false;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;

    private TextToSpeech textToSpeech;
    private SpeechRecognizer speechRecognizer;
    private boolean isListeningForStop = false;
    private final Handler voiceHandler = new Handler();
    private long lastSpeakTime = 0;
    private static final long SPEAK_INTERVAL_MS = 3000;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private Integer sensorOrientation;

    @Override
    public synchronized void onResume() {
        super.onResume();
        initVoiceServices();
    }

    @Override
    public synchronized void onPause() {
        shutdownVoiceServices();
        super.onPause();
    }

    private void initVoiceServices() {
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.US);
            }
        });
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle params) {}
            @Override
            public void onBeginningOfSpeech() { isListeningForStop = true; }
            @Override
            public void onRmsChanged(float rmsdB) {}
            @Override
            public void onBufferReceived(byte[] buffer) {}
            @Override
            public void onEndOfSpeech() { isListeningForStop = false; }
            @Override
            public void onError(int error) {
                isListeningForStop = false;
                if (!isFinishing()) startStopListener();
            }
            @Override
            public void onResults(Bundle results) {
                isListeningForStop = false;
                ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null) {
                    for (String match : matches) {
                        String cmd = match.toLowerCase();
                        if (cmd.contains("help") || cmd.contains("मदद") || cmd.contains("సహాయం")) {
                            fireEmergencyIntent();
                            return;
                        }
                        if (cmd.contains("stop") || cmd.contains("बंद") || cmd.contains("ఆపు")) {
                            stopDetection();
                            return;
                        }
                    }
                }
                if (!isFinishing()) startStopListener();
            }
            @Override
            public void onPartialResults(Bundle partialResults) {}
            @Override
            public void onEvent(int eventType, Bundle params) {}
        });
        startStopListener();
    }

    private void shutdownVoiceServices() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
        voiceHandler.removeCallbacksAndMessages(null);
    }

    private void startStopListener() {
        if (!isListeningForStop && !isFinishing()) {
            isListeningForStop = true;
            voiceHandler.post(() -> {
                try {
                    Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                    intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                    speechRecognizer.startListening(intent);
                } catch (Exception e) {
                    isListeningForStop = false;
                    Log.e(TAG, "Failed to start stop listener", e);
                }
            });
        }
    }

    private void fireEmergencyIntent() {
        Intent intent = new Intent(this, HomeActivity.class);
        intent.setAction(HomeActivity.ACTION_EMERGENCY);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
        finish();
    }

    private void stopDetection() {
        if (textToSpeech != null) {
            textToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                @Override public void onStart(String utteranceId) {}
                @Override public void onDone(String utteranceId) { finish(); }
                @Override public void onError(String utteranceId) { finish(); }
            });
            textToSpeech.speak("Stopping detection", TextToSpeech.QUEUE_FLUSH, null, "StopId");
        } else {
            finish();
        }
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        try {
            detector = DetectorFactory.getDetector(getAssets(), "yolov5s.tflite");
        } catch (final IOException e) {
            e.printStackTrace();
            Toast.makeText(getApplicationContext(), "Classifier could not be initialized.", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }
        int cropSize = detector.getInputSize();
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();
        sensorOrientation = rotation - getScreenOrientation();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);
        frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, true);
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
        tracker = new MultiBoxTracker(this);
        
        OverlayView trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    tracker.draw(canvas);
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage() {
        if (computingDetection || rgbFrameBitmap == null) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        runInBackground(() -> {
            try {
                final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                final List<Classifier.Recognition> mappedRecognitions = new LinkedList<>();
                Classifier.Recognition bestMatch = null;

                for (final Classifier.Recognition result : results) {
                    final RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                        cropToFrameTransform.mapRect(location);
                        result.setLocation(location);
                        mappedRecognitions.add(result);
                        if (result.getDistance() != null) {
                            if (bestMatch == null || result.getDistance() < bestMatch.getDistance()) {
                                bestMatch = result;
                            }
                        }
                    }
                }
                tracker.trackResults(mappedRecognitions, System.currentTimeMillis());
                if (bestMatch != null) {
                    final Classifier.Recognition finalBestMatch = bestMatch;
                    runOnUiThread(() -> announceObject(finalBestMatch));
                }
            } catch (Exception e) {
                Log.e(TAG, "Exception in detection background thread!", e);
            }
            computingDetection = false;
        });
    }

    private void announceObject(Classifier.Recognition recognition) {
        long currentTime = System.currentTimeMillis();
        if (textToSpeech != null && !textToSpeech.isSpeaking() && (currentTime - lastSpeakTime > SPEAK_INTERVAL_MS)) {
            String message = recognition.getTitle() + ", " + String.format(Locale.US, "%.1f meters", recognition.getDistance());
            textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, null);
            lastSpeakTime = currentTime;
        }
    }

    @Override
    protected int getLayoutId() { return R.layout.tfe_od_camera_connection_fragment_tracking; }
    @Override
    protected Size getDesiredPreviewFrameSize() { return DESIRED_PREVIEW_SIZE; }
    
    @Override
    protected void setUseNNAPI(boolean isChecked) {
        if(detector != null) runInBackground(() -> detector.setUseNNAPI(isChecked));
    }
    @Override
    protected void setNumThreads(int numThreads) {
        if(detector != null) runInBackground(() -> detector.setNumThreads(numThreads));
    }

    // This method is required by CameraActivity but not used in our voice-only app
    @Override
    protected void updateActiveModel() {}
}
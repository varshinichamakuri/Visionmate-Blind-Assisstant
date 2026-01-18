package org.tensorflow.lite.examples.detection;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;
import android.util.SparseArray;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.vision.CameraSource;
import com.google.android.gms.vision.Detector;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;

public class OCRActivity extends AppCompatActivity {

    private static final String TAG = "OCRActivity";

    private CameraSource mCameraSource;
    private TextRecognizer mTextRecognizer;
    private SurfaceView mSurfaceView;
    private TextView mTextView;

    private TextToSpeech textToSpeech;
    private SpeechRecognizer speechRecognizer;
    private static final int RC_HANDLE_CAMERA_PERM = 2;
    private boolean isListeningForStop = false;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private String lastSpokenText = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ocractivity);
        mSurfaceView = findViewById(R.id.surfaceView);
        mTextView = findViewById(R.id.textView);
    }

    @Override
    protected void onResume() {
        super.onResume();
        initVoiceServices();
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startTextRecognizer();
        } else {
            askCameraPermission();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mCameraSource != null) {
            try {
                mCameraSource.stop();
            } catch (Exception e) {
                Log.e(TAG, "Error stopping camera source", e);
            }
        }
        shutdownVoiceServices();
    }

    private void initVoiceServices() {
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.ENGLISH);
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
                            stopReading();
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
        handler.removeCallbacksAndMessages(null);
    }

    private void startStopListener() {
        if (!isListeningForStop && !isFinishing()) {
            isListeningForStop = true;
            handler.post(() -> {
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

    private void stopReading() {
        if (textToSpeech != null) {
            textToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                 @Override
                 public void onStart(String utteranceId) {}
                 @Override
                 public void onDone(String utteranceId) {
                     finish();
                 }
                 @Override
                 public void onError(String utteranceId) {
                     finish();
                 }
             });
            textToSpeech.speak("Stopping text reader", TextToSpeech.QUEUE_FLUSH, null, "StopReadingId");
        } else {
            finish();
        }
    }

    private void startTextRecognizer() {
        mTextRecognizer = new TextRecognizer.Builder(getApplicationContext()).build();
        if (!mTextRecognizer.isOperational()) {
            Toast.makeText(getApplicationContext(), "Could not set up the text recognizer.", Toast.LENGTH_LONG).show();
            return;
        }
        mCameraSource = new CameraSource.Builder(getApplicationContext(), mTextRecognizer)
                .setFacing(CameraSource.CAMERA_FACING_BACK)
                .setRequestedPreviewSize(1280, 1024)
                .setRequestedFps(15.0f)
                .setAutoFocusEnabled(true)
                .build();

        mSurfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                try {
                    if (ActivityCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                        mCameraSource.start(mSurfaceView.getHolder());
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            @Override
            public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {}
            @Override
            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
                if (mCameraSource != null) {
                    try { mCameraSource.stop(); } catch (Exception e) {}
                }
            }
        });

        mTextRecognizer.setProcessor(new Detector.Processor<TextBlock>() {
            @Override
            public void release() {}

            @Override
            public void receiveDetections(@NonNull Detector.Detections<TextBlock> detections) {
                final SparseArray<TextBlock> items = detections.getDetectedItems();
                if (items.size() != 0) {
                    mTextView.post(() -> {
                        StringBuilder stringBuilder = new StringBuilder();
                        for (int i = 0; i < items.size(); ++i) {
                            TextBlock item = items.valueAt(i);
                            if (item != null && item.getValue() != null) {
                                stringBuilder.append(item.getValue()).append(" ");
                            }
                        }
                        String detectedText = stringBuilder.toString();
                        mTextView.setText(detectedText);
                        
                        if (textToSpeech != null && !textToSpeech.isSpeaking() && !detectedText.isEmpty() && isNewText(detectedText)) {
                            lastSpokenText = detectedText;
                            textToSpeech.speak(detectedText, TextToSpeech.QUEUE_FLUSH, null, null);
                        }
                    });
                }
            }
        });
    }

    private boolean isNewText(String newText) {
        if (lastSpokenText.isEmpty()) return true;
        return Math.abs(newText.length() - lastSpokenText.length()) > lastSpokenText.length() * 0.1 || 
               !newText.substring(0, Math.min(newText.length(), 10)).equals(lastSpokenText.substring(0, Math.min(lastSpokenText.length(), 10)));
    }

    private void askCameraPermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, RC_HANDLE_CAMERA_PERM);
    }
}

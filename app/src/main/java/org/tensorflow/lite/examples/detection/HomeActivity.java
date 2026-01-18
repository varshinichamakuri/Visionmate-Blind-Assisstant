package org.tensorflow.lite.examples.detection;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.location.Location;
import android.location.LocationManager;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.telephony.SmsManager;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class HomeActivity extends CameraActivity implements ImageReader.OnImageAvailableListener {

    private static final String TAG = "HomeActivity";

    private TextToSpeech textToSpeech;
    private SpeechRecognizer speechRecognizer;
    private static final int REQUEST_CODE_PERMISSIONS = 101;
    private final Handler handler = new Handler(Looper.getMainLooper());

    // State Management
    private enum AppState { IDLE, LISTENING, DETECTING, READING, EMERGENCY }
    private AppState currentState = AppState.IDLE;

    private LocationManager locationManager;

    // --- Language & Command Management ---
    private static final String LANG_ENGLISH = "en";
    private static final String LANG_HINDI = "hi";
    private static final String LANG_TELUGU = "te";
    private String currentLanguage = LANG_ENGLISH;
    private Map<String, List<String>> commandMap;

    private static final String EMERGENCY_NUMBER = "1234567890";

    // --- Detection Fields ---
    private Classifier detector;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private boolean computingDetection = false;
    private Matrix frameToCropTransform;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private long lastSpeakTime = 0;
    private static final long SPEAK_INTERVAL_MS = 3000;
    private Integer sensorOrientation;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        initializeCommands();
        locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
        checkPermissionAndStart();
    }

    private void initializeCommands() {
        commandMap = new HashMap<>();
        commandMap.put(LANG_ENGLISH, Arrays.asList("detect", "read", "help"));
        commandMap.put(LANG_HINDI, Arrays.asList("पहचानो", "पढ़ो", "मदद"));
        commandMap.put(LANG_TELUGU, Arrays.asList("గుర్తించు", "చదవు", "సహాయం"));
    }

    @Override
    public synchronized void onResume() {
        super.onResume();
        enterIdleState();
    }

    @Override
    public synchronized void onPause() {
        if (currentState == AppState.DETECTING) {
            stopDetection();
        }
        stopListening();
        super.onPause();
    }

    private void checkPermissionAndStart() {
        String[] permissions = { Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO, Manifest.permission.CALL_PHONE, 
                                 Manifest.permission.SEND_SMS, Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION };
        if (!hasPermissions(this, permissions)) {
            ActivityCompat.requestPermissions(this, permissions, REQUEST_CODE_PERMISSIONS);
        } else {
            setFragment();
            initVoiceFeatures();
        }
    }

    private boolean hasPermissions(Context context, String... permissions) {
        for (String permission : permissions) {
            if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS && hasPermissions(this, permissions)) {
            setFragment();
            initVoiceFeatures();
        } else {
            Toast.makeText(this, "Permissions Denied. App cannot function.", Toast.LENGTH_LONG).show();
            finish();
        }
    }

    private void initVoiceFeatures() {
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                setTTSLanguage(currentLanguage);
                textToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override
                    public void onStart(String utteranceId) {
                        runOnUiThread(() -> stopListening());
                    }
                    @Override
                    public void onDone(String utteranceId) {
                        if (utteranceId.startsWith("PROMPT")) {
                            runOnUiThread(() -> startListening());
                        }
                    }
                    @Override
                    public void onError(String utteranceId) {
                        runOnUiThread(() -> startListening());
                    }
                });
                enterIdleState();
            } else {
                Log.e(TAG, "TTS Initialization failed");
            }
        });

        initializeSpeechRecognizer();
    }

    private void enterIdleState() {
        if (currentState == AppState.DETECTING) {
            stopDetection();
        }
        currentState = AppState.IDLE;
        speak(getLocalizedPrompt(), "PROMPT_IDLE");
    }

    private void setTTSLanguage(String langCode) {
        Locale locale = langCode.equals(LANG_HINDI) ? new Locale("hi", "IN") : 
                        langCode.equals(LANG_TELUGU) ? new Locale("te", "IN") : Locale.ENGLISH;
        if (textToSpeech != null) {
            int result = textToSpeech.setLanguage(locale);
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                if (!langCode.equals(LANG_ENGLISH)) {
                    currentLanguage = LANG_ENGLISH;
                    textToSpeech.setLanguage(Locale.ENGLISH);
                }
            }
        }
    }

    private String getLocalizedPrompt() {
        switch (currentLanguage) {
            case LANG_TELUGU: return "మీకు ఏమి కావాలి?";
            case LANG_HINDI: return "आपको क्या चाहिए?";
            default: return "What do you need?";
        }
    }

    private void initializeSpeechRecognizer() {
        if (SpeechRecognizer.isRecognitionAvailable(this)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
            speechRecognizer.setRecognitionListener(new RecognitionListener() {
                @Override
                public void onReadyForSpeech(Bundle params) { }
                @Override
                public void onBeginningOfSpeech() { currentState = AppState.LISTENING; }
                @Override
                public void onRmsChanged(float rmsdB) { }
                @Override
                public void onBufferReceived(byte[] buffer) { }
                @Override
                public void onEndOfSpeech() { currentState = AppState.IDLE; }
                @Override
                public void onError(int error) { enterIdleState(); }
                @Override
                public void onResults(Bundle results) {
                    ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                    if (matches != null && !matches.isEmpty()) {
                        processCommand(matches.get(0));
                    } else {
                        enterIdleState();
                    }
                }
                @Override
                public void onPartialResults(Bundle partialResults) { }
                @Override
                public void onEvent(int eventType, Bundle params) { }
            });
        }
    }

    private void speak(String text, String utteranceId) {
        if (textToSpeech != null) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId);
        }
    }

    private void startListening() {
        if (speechRecognizer != null && currentState == AppState.IDLE) {
            try {
                Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                String langTag = currentLanguage.equals(LANG_HINDI) ? "hi-IN" : currentLanguage.equals(LANG_TELUGU) ? "te-IN" : "en-IN";
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, langTag);
                speechRecognizer.startListening(intent);
            } catch (Exception e) {
                Log.e(TAG, "startListening failed", e);
            }
        }
    }

    private void stopListening() {
        if (speechRecognizer != null) {
            try { speechRecognizer.stopListening(); } catch(Exception e) {}
        }
    }

    private void processCommand(String command) {
        if (currentState != AppState.IDLE) return;
        String lowerCommand = command.toLowerCase();

        if (lowerCommand.contains("hindi")) {
            changeLanguage(LANG_HINDI, "भाषा हिंदी में बदल दी गई है", "आपको क्या चाहिए?");
            return;
        } else if (lowerCommand.contains("telugu")) {
            changeLanguage(LANG_TELUGU, "భాష తెలుగులోకి మార్చబడింది", "మీకు ఏమి కావాలి?");
            return;
        } else if (lowerCommand.contains("english")) {
            changeLanguage(LANG_ENGLISH, "Language changed to English", "What do you need?");
            return;
        }

        List<String> commands = commandMap.get(currentLanguage);
        if (lowerCommand.contains(commands.get(0))) { // Detect
            startDetection();
        } else if (lowerCommand.contains(commands.get(1))) { // Read
            speak("Text reading is not yet implemented.", "INFO");
        } else if (lowerCommand.contains(commands.get(2))) { // Help
            triggerEmergencySequence();
        } else if(lowerCommand.contains("stop") && currentState == AppState.DETECTING) {
            enterIdleState();
        } else {
            enterIdleState();
        }
    }

    private void changeLanguage(String langCode, String confirmationMsg, String nextPrompt) {
        currentLanguage = langCode;
        setTTSLanguage(langCode);
        textToSpeech.speak(confirmationMsg, TextToSpeech.QUEUE_FLUSH, null, "CONFIRM_LANG");
        textToSpeech.speak(nextPrompt, TextToSpeech.QUEUE_ADD, null, "PROMPT_NEW_LANG");
    }
    
    private void startDetection() {
        currentState = AppState.DETECTING;
        speak("Detection mode started. Say stop to exit.", "START_DETECT");
    }

    private void stopDetection() {
        computingDetection = false;
        currentState = AppState.IDLE;
    }

    private void triggerEmergencySequence() {
        if (currentState == AppState.EMERGENCY) return;
        currentState = AppState.EMERGENCY;
        stopListening();
        String msg = getLocalizedEmergencyMessage("Emergency activated. Calling now.");
        speak(msg, "EMERGENCY_CALL");
        vibrate();
        handler.postDelayed(this::performEmergencyActions, 4000);
    }

    private void performEmergencyActions() {
        if (isFinishing()) return;
        sendEmergencyLocationSMS();
        makePhoneCall();
    }
    
    // --- CameraActivity Implementation ---
    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        try {
            detector = DetectorFactory.getDetector(getAssets(), "yolov5s.tflite");
        } catch (final IOException e) {
            Log.e(TAG, "Classifier could not be initialized", e);
            finish();
            return;
        }
        int cropSize = detector.getInputSize();
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();
        sensorOrientation = rotation - getScreenOrientation();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);
        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, true);
    }

    @Override
    protected void processImage() {
        if (currentState != AppState.DETECTING || computingDetection || rgbFrameBitmap == null) {
            readyForNextImage();
            return;
        }
        computingDetection = true;

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        runInBackground(
                () -> {
                    try {
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        Classifier.Recognition bestMatch = null;

                        for (final Classifier.Recognition result : results) {
                            if (result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API && result.getDistance() != null) {
                                if (bestMatch == null || result.getDistance() < bestMatch.getDistance()) {
                                    bestMatch = result;
                                }
                            }
                        }

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
            textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, "OBJECT");
            lastSpeakTime = currentTime;
        }
    }
    
    @Override
    protected int getLayoutId() { return R.layout.tfe_od_camera_connection_fragment_tracking; }

    @Override
    protected Size getDesiredPreviewFrameSize() { return DESIRED_PREVIEW_SIZE; }
    
    @Override
    protected void onInferenceConfigurationChanged() {}

    // Helper methods for emergency actions (vibrate, sms, call)
    private void vibrate() {
        Vibrator v = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        if (v != null) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                v.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE));
            } else {
                v.vibrate(500);
            }
        }
    }

    private void sendEmergencyLocationSMS() {
        if (!hasPermissions(this, Manifest.permission.SEND_SMS, Manifest.permission.ACCESS_FINE_LOCATION)) return;
        try {
            Location location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
            if (location == null) location = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
            
            String msg = "EMERGENCY! My location: https://maps.google.com/?q=" + 
                         (location != null ? location.getLatitude() + "," + location.getLongitude() : "unavailable");
            SmsManager.getDefault().sendTextMessage(EMERGENCY_NUMBER, null, msg, null, null);
        } catch (Exception e) {
            Log.e(TAG, "SMS failed", e);
        }
    }

    private void makePhoneCall() {
        try {
            if (hasPermissions(this, Manifest.permission.CALL_PHONE)) {
                startActivity(new Intent(Intent.ACTION_CALL, Uri.parse("tel:" + EMERGENCY_NUMBER)));
            } else {
                startActivity(new Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + EMERGENCY_NUMBER)));
            }
        } catch (Exception e) {
            Log.e(TAG, "Call failed", e);
            enterIdleState();
        }
    }

    private String getLocalizedEmergencyMessage(String defaultMsg) {
        switch (currentLanguage) {
            case LANG_TELUGU: return "అత్యవసర సహాయం కాల్ చేయబడుతోంది";
            case LANG_HINDI: return "आपातकालीन सहायता सक्रिय। अभी कॉल कर रहे हैं।";
            default: return defaultMsg;
        }
    }
    
    // Unused sensor methods
    @Override
    public void onSensorChanged(android.hardware.SensorEvent event) {}
    @Override
    public void onAccuracyChanged(android.hardware.Sensor sensor, int accuracy) {}
    @Override
    protected void setUseNNAPI(boolean isChecked) {}
    @Override
    protected void setNumThreads(int numThreads) {}
    @Override
    protected void updateActiveModel() {}
}
package org.tensorflow.lite.examples.detection;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Location;
import android.location.LocationManager;
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
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Locale;

public class HomeActivity extends AppCompatActivity implements SensorEventListener {

    private static final String TAG = "HomeActivity";
    private Button camera_button;
    private Button ocr_button;

    private TextToSpeech textToSpeech;
    private SpeechRecognizer speechRecognizer;
    private static final int REQUEST_CODE_PERMISSIONS = 101;
    private boolean isListening = false;
    private boolean isProcessingCommand = false;
    private final Handler handler = new Handler(Looper.getMainLooper());

    private SensorManager sensorManager;
    private Sensor accelerometer;
    private float mAccel;
    private float mAccelCurrent;
    private float mAccelLast;
    private boolean isEmergencyTriggered = false;
    private LocationManager locationManager;

    // Language constants
    private static final String LANG_ENGLISH = "en";
    private static final String LANG_TELUGU = "te";
    private static final String LANG_HINDI = "hi";
    private String currentLanguage = LANG_ENGLISH; // Default language

    // Emergency Contact - Updated to mock testing number
    private static final String EMERGENCY_NUMBER = "1234567890"; 

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        camera_button = findViewById(R.id.cameraButton);
        ocr_button = findViewById(R.id.OCR);

        camera_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(HomeActivity.this, DetectorActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP));
            }
        });
        ocr_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(HomeActivity.this, OCRActivity.class));
            }
        });

        // Initialize Sensor Manager
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            mAccel = 0.00f;
            mAccelCurrent = SensorManager.GRAVITY_EARTH;
            mAccelLast = SensorManager.GRAVITY_EARTH;
        }

        locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);

        // Permissions check
        checkPermissionAndStart();
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        if (sensorManager != null && accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);
        }
        // Restart listening if we are not busy and not in emergency mode
        if (textToSpeech != null && !isListening && !isProcessingCommand && !isEmergencyTriggered) {
            startListening();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
        stopListening();
    }

    private void checkPermissionAndStart() {
        String[] permissions = {
            Manifest.permission.RECORD_AUDIO, 
            Manifest.permission.CALL_PHONE,
            Manifest.permission.SEND_SMS,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        };
        boolean allGranted = true;
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }

        if (!allGranted) {
            ActivityCompat.requestPermissions(this, permissions, REQUEST_CODE_PERMISSIONS);
        } else {
            initVoiceFeatures();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            boolean allGranted = true;
            for(int res : grantResults){
                if(res != PackageManager.PERMISSION_GRANTED){
                    allGranted = false;
                    break;
                }
            }
            if (allGranted) {
                initVoiceFeatures();
            } else {
                Toast.makeText(this, "Permissions Denied. Some features will not work.", Toast.LENGTH_SHORT).show();
                // Still try to init voice for basic features if AUDIO is granted
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    initVoiceFeatures();
                }
            }
        }
    }

    private void initVoiceFeatures() {
        textToSpeech = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    setTTSLanguage(currentLanguage);
                    
                    textToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                        @Override
                        public void onStart(String utteranceId) {
                            // Stop listening while speaking to avoid hearing itself
                            runOnUiThread(() -> stopListening());
                        }

                        @Override
                        public void onDone(String utteranceId) {
                            runOnUiThread(() -> {
                                // Only restart listening if we are not in the middle of an emergency action
                                if (!isEmergencyTriggered) {
                                    startListening();
                                }
                            });
                        }

                        @Override
                        public void onError(String utteranceId) {
                            runOnUiThread(() -> {
                                if (!isEmergencyTriggered) {
                                    startListening();
                                }
                            });
                        }
                    });
                    
                    speak(getLocalizedPrompt());
                } else {
                    Log.e(TAG, "TTS Initialization failed");
                }
            }
        });

        initializeSpeechRecognizer();
    }
    
    private void setTTSLanguage(String langCode) {
        Locale locale = new Locale(langCode);
        if (langCode.equals(LANG_HINDI)) {
            locale = new Locale("hi", "IN");
        } else if (langCode.equals(LANG_TELUGU)) {
            locale = new Locale("te", "IN");
        }
        
        int result = textToSpeech.setLanguage(locale);
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            Toast.makeText(HomeActivity.this, "Language not supported", Toast.LENGTH_SHORT).show();
            // Fallback to English
            if (!langCode.equals(LANG_ENGLISH)) {
                textToSpeech.setLanguage(Locale.US);
                currentLanguage = LANG_ENGLISH;
            }
        }
    }

    private String getLocalizedPrompt() {
        switch (currentLanguage) {
            case LANG_TELUGU:
                return "మీరు ఏమి చేయాలనుకుంటున్నారు?";
            case LANG_HINDI:
                return "आप क्या करना चाहते हैं?";
            default:
                return "What do you need?";
        }
    }

    private void initializeSpeechRecognizer() {
        if (SpeechRecognizer.isRecognitionAvailable(this)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
            speechRecognizer.setRecognitionListener(new RecognitionListener() {
                @Override
                public void onReadyForSpeech(Bundle params) { }

                @Override
                public void onBeginningOfSpeech() {
                    isListening = true;
                }

                @Override
                public void onRmsChanged(float rmsdB) { }

                @Override
                public void onBufferReceived(byte[] buffer) { }

                @Override
                public void onEndOfSpeech() {
                    isListening = false;
                }

                @Override
                public void onError(int error) {
                    isListening = false;
                    // Restart listening after a short delay on error
                    if (!isEmergencyTriggered && !isProcessingCommand) {
                        restartListeningWithDelay();
                    }
                }

                @Override
                public void onResults(Bundle results) {
                    isListening = false;
                    ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                    if (matches != null && !matches.isEmpty()) {
                        processCommand(matches.get(0));
                    } else {
                        if (!isEmergencyTriggered && !isProcessingCommand) {
                            restartListeningWithDelay();
                        }
                    }
                }

                @Override
                public void onPartialResults(Bundle partialResults) { }

                @Override
                public void onEvent(int eventType, Bundle params) { }
            });
        }
    }

    private void speak(String text) {
        if (textToSpeech != null) {
            Bundle params = new Bundle();
            params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "utteranceId");
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, params, "utteranceId");
        }
    }

    private void startListening() {
        if (speechRecognizer != null && !isListening && !isProcessingCommand && !isEmergencyTriggered) {
            try {
                Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                
                String langTag = currentLanguage;
                if (currentLanguage.equals(LANG_HINDI)) langTag = "hi-IN";
                else if (currentLanguage.equals(LANG_TELUGU)) langTag = "te-IN";
                else langTag = "en-US";
                
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, langTag);
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, langTag); 
                intent.putExtra(RecognizerIntent.EXTRA_ONLY_RETURN_LANGUAGE_PREFERENCE, langTag);
                
                speechRecognizer.startListening(intent);
                isListening = true;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void stopListening() {
        if (speechRecognizer != null) {
            try {
                speechRecognizer.stopListening();
            } catch (Exception e) {
                e.printStackTrace();
            }
            isListening = false;
        }
    }
    
    private void restartListeningWithDelay() {
        handler.postDelayed(() -> {
            if (!isProcessingCommand && !isEmergencyTriggered && !isFinishing()) {
               startListening();
            }
        }, 1000); // Wait 1 second before restarting
    }

    private void processCommand(String command) {
        if (isProcessingCommand || isEmergencyTriggered) return;
        
        String lowerCommand = command.toLowerCase();
        
        // Language Change Commands
        if (lowerCommand.contains("language change to telugu") || lowerCommand.contains("change language to telugu")) {
            changeLanguage(LANG_TELUGU, "భాష తెలుగుకు మార్చబడింది");
            return;
        } else if (lowerCommand.contains("language change to hindi") || lowerCommand.contains("change language to hindi")) {
            changeLanguage(LANG_HINDI, "भाषा हिंदी में बदल दी गई है");
            return;
        } else if (lowerCommand.contains("language change to english") || lowerCommand.contains("change language to english")) {
            changeLanguage(LANG_ENGLISH, "Language changed to English");
            return;
        }

        // Emergency
        if (lowerCommand.contains("help") || lowerCommand.contains("సహాయం") || lowerCommand.contains("मदद") || lowerCommand.contains("बचाओ")) {
            triggerEmergencyCall();
        } 
        // Detect / Camera
        else if (lowerCommand.contains("detect") || lowerCommand.contains("object") || lowerCommand.contains("gurthinchu") || lowerCommand.contains("pehchano") || lowerCommand.contains("dekho")) {
            isProcessingCommand = true;
            startActivity(new Intent(HomeActivity.this, DetectorActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP));
            // Only reset flag, don't auto-restart listening here as activity changes
            handler.postDelayed(() -> isProcessingCommand = false, 2000);
        } 
        // OCR / Read
        else if (lowerCommand.contains("ocr") || lowerCommand.contains("read") || lowerCommand.contains("text") || lowerCommand.contains("chaduvu") || lowerCommand.contains("padho")) {
             isProcessingCommand = true;
            startActivity(new Intent(HomeActivity.this, OCRActivity.class));
            handler.postDelayed(() -> isProcessingCommand = false, 2000);
        } 
        // Info / Help
        else if (lowerCommand.contains("info")) { 
             speak(getLocalizedPrompt());
        } else {
             // Not recognized, restart listening
             restartListeningWithDelay();
        }
    }
    
    private void changeLanguage(String langCode, String confirmationMsg) {
        currentLanguage = langCode;
        setTTSLanguage(langCode);
        speak(confirmationMsg);
    }

    // --- Emergency Feature ---

    private void triggerEmergencyCall() {
        if (isEmergencyTriggered) return;
        isEmergencyTriggered = true;
        
        // 1. Immediately stop voice recognition to prevent overlaps/crashes
        stopListening();
        
        // 2. Speak confirmation
        String msg = "Emergency help activated. Calling now.";
        if (currentLanguage.equals(LANG_TELUGU)) msg = "అత్యవసర సేవలు ప్రారంభించబడ్డాయి. ఇప్పుడే కాల్ చేస్తున్నారు.";
        else if (currentLanguage.equals(LANG_HINDI)) msg = "आपातकालीन सहायता सक्रिय। अभी कॉल कर रहे हैं।";
        
        speak(msg);
        vibrate();
        
        // 3. Delay emergency actions to allow TTS to finish speaking (safe delay)
        handler.postDelayed(() -> performEmergencyActions(), 4000);
    }

    private void performEmergencyActions() {
        if (isFinishing()) return;
        
        // Send SMS first (background operation)
        sendEmergencyLocationSMS();
        
        // Make the call (foreground operation - might pause/stop activity)
        makePhoneCall();
        
        // Reset emergency state after a sufficient delay to prevent loops
        handler.postDelayed(() -> isEmergencyTriggered = false, 15000);
    }
    
    private void vibrate() {
        try {
            Vibrator v = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
            if (v != null) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    v.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE));
                } else {
                    v.vibrate(500);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Vibration failed", e);
        }
    }

    private void sendEmergencyLocationSMS() {
        // Safe check for permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "Permissions missing for SMS/Location");
            return;
        }

        try {
            Location location = null;
            if (locationManager != null) {
                // Try GPS first
                location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
                // Fallback to Network
                if (location == null) {
                    location = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
                }
            }
            
            String msg = "EMERGENCY! I need help. My location: ";
            if (location != null) {
                msg += "https://maps.google.com/?q=" + location.getLatitude() + "," + location.getLongitude();
            } else {
                msg += "Location unavailable.";
            }

            SmsManager smsManager = SmsManager.getDefault();
            smsManager.sendTextMessage(EMERGENCY_NUMBER, null, msg, null, null);
            Toast.makeText(this, "Emergency SMS Sent", Toast.LENGTH_SHORT).show();

        } catch (SecurityException e) {
            Log.e(TAG, "Security Exception during SMS/Location", e);
        } catch (Exception e) {
            Log.e(TAG, "Failed to send SMS", e);
        }
    }
    
    private void makePhoneCall() {
        try {
            // Check Call Permission safely
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE) == PackageManager.PERMISSION_GRANTED) {
                Intent callIntent = new Intent(Intent.ACTION_CALL);
                callIntent.setData(Uri.parse("tel:" + EMERGENCY_NUMBER));
                startActivity(callIntent);
            } else {
                // Fallback to ACTION_DIAL if permission is missing (User has to press call)
                Intent dialIntent = new Intent(Intent.ACTION_DIAL);
                dialIntent.setData(Uri.parse("tel:" + EMERGENCY_NUMBER));
                startActivity(dialIntent);
            }
        } catch (Exception e) {
            Log.e(TAG, "Call failed", e);
            Toast.makeText(this, "Call Failed", Toast.LENGTH_SHORT).show();
            // In case of failure, reset trigger so user can try again
            isEmergencyTriggered = false;
        }
    }

    // --- Shake Detection ---

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (isEmergencyTriggered) return;

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            float x = event.values[0];
            float y = event.values[1];
            float z = event.values[2];
            mAccelLast = mAccelCurrent;
            mAccelCurrent = (float) Math.sqrt((double) (x * x + y * y + z * z));
            float delta = mAccelCurrent - mAccelLast;
            mAccel = mAccel * 0.9f + delta; // Perform low-cut filter

            if (mAccel > 12) { // Sensitivity threshold
                triggerEmergencyShake();
            }
        }
    }
    
    private void triggerEmergencyShake() {
        if (isEmergencyTriggered) return;
        isEmergencyTriggered = true;
        stopListening();
        
        String msg = "Emergency detected, calling now";
        if (currentLanguage.equals(LANG_TELUGU)) msg = "అత్యవసర పరిస్థితి గుర్తించబడింది, ఇప్పుడే కాల్ చేస్తున్నారు";
        else if (currentLanguage.equals(LANG_HINDI)) msg = "आपातकालीन स्थिति का पता चला, अभी कॉल कर रहे हैं";

        speak(msg);
        vibrate();
        
        handler.postDelayed(() -> performEmergencyActions(), 4000);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    @Override
    protected void onDestroy() {
        // Lifecycle safety: Release resources to avoid leaks and crashes
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        if (speechRecognizer != null) {
            try {
                speechRecognizer.destroy();
            } catch (Exception e) {
                // ignore
            }
        }
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
        // Remove any pending posts to handler to avoid memory leaks or crashes after destroy
        handler.removeCallbacksAndMessages(null);
        super.onDestroy();
    }
}
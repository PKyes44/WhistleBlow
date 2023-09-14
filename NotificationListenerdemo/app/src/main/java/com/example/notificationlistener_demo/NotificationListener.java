package com.example.notificationlistener_demo;

import android.app.Notification;
import android.app.PendingIntent;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.service.notification.NotificationListenerService;
import android.service.notification.StatusBarNotification;
import android.telephony.SmsManager;
import android.text.TextUtils;
import android.util.Log;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

public class NotificationListener extends NotificationListenerService {

    private String TAG = "NotificationListenerService";
    private final int SEND_WEB_RETURN = 200;

    @Override
    public void onNotificationPosted(StatusBarNotification sbn) {

        String packageName = sbn.getPackageName();
        Bundle extras = sbn.getNotification().extras;

        String extraBigText = "";
        String extraInfoText = "";
        String extraSubText = "";
        String extraSummaryText = "";
        String extraTitle = "";
        String extraText = "";
        if (extras.get(Notification.EXTRA_TITLE) != null) {
            extraTitle = extras.get(Notification.EXTRA_TITLE).toString();
        }
        if (extras.get(Notification.EXTRA_TEXT) != null) {
            extraText = extras.get(Notification.EXTRA_TEXT).toString();
        }
        if (extras.get(Notification.EXTRA_BIG_TEXT) != null) {
            extraBigText = extras.get(Notification.EXTRA_BIG_TEXT).toString();
        }
        if (extras.get(Notification.EXTRA_INFO_TEXT) != null) {
            extraInfoText = extras.get(Notification.EXTRA_INFO_TEXT).toString();
        }
        if (extras.get(Notification.EXTRA_SUB_TEXT) != null) {
            extraSubText = extras.get(Notification.EXTRA_SUB_TEXT).toString();
        }
        if (extras.get(Notification.EXTRA_SUMMARY_TEXT) != null) {
            extraSummaryText = extras.get(Notification.EXTRA_SUMMARY_TEXT).toString();
        }
        if (Objects.equals(packageName, "com.instagram.android")) {
            Log.d(
                    TAG, "onNotificationPosted:\n" +
                            "PackageName: " + packageName + "\n" +
                            "Title: " + extraTitle + "\n" +
                            "Text: " + extraText + "\n" +
                            "BigText: " + extraBigText + "\n" +
                            "InfoText: " + extraInfoText + "\n" +
                            "SubText: " + extraSubText + "\n" +
                            "SummaryText: " + extraSummaryText + "\n"
            );

            ContentValues addRowValue = new ContentValues();

            addRowValue.put("comment", extraText);
            addRowValue.put("sender", extraTitle);
            addRowValue.put("regDate", getNowTime());

            NetworkTask HTTP_REQ = new NetworkTask(addRowValue);
            HTTP_REQ.execute();
        }
    }

    private String getNowTime() {
        long now = System.currentTimeMillis();
        Date mDate = new Date(now);

        SimpleDateFormat sdfNow = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String time = sdfNow.format(mDate);

        return time;
    }
}

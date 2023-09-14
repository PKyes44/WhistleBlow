package com.example.notificationlistener_demo;

import android.content.ContentValues;
import android.os.AsyncTask;
import android.os.Handler;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

public class NetworkTask extends AsyncTask<Void, Void, String> {

    private ContentValues values;
    private DatabaseReference mDatabase;
    private int pushCount;
    ArrayList<Push> pushList = new ArrayList<Push>();

    public NetworkTask(ContentValues values) {
        this.values = values;
    }

    @Override
    protected String doInBackground(Void... params) {

        String result; // 요청 결과를 저장할 변수.
        RequestHttpURLConnection requestHttpURLConnection = new RequestHttpURLConnection();
        result = requestHttpURLConnection.request(values); // 해당 URL로 부터 결과물을 얻어온다.

        return result;
    }

    @Override
    protected void onPostExecute(String s) {
        super.onPostExecute(s);
        mDatabase = FirebaseDatabase.getInstance().getReference();
        System.out.println("s = " + s);
    }



    private void writePush(int pushId,
//                           String sender, String content, String regDate,
                           String s
    ) {
//        Push push = new Push(sender, content, regDate);
        mDatabase.child("push").child(String.valueOf(pushId)).setValue(s)
                .addOnSuccessListener(new OnSuccessListener<Void>() {
                    @Override
                    public void onSuccess(Void unused) {
                        System.out.println("PushUpload_Clear");
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        System.out.println("PushUpload_FAIL");
                    }
                });
        mDatabase.child("push")
                .addChildEventListener(new ChildEventListener() {
                    @Override
                    public void onChildAdded(@NonNull DataSnapshot snapshot, @Nullable String previousChildName) {
                        mDatabase.child("pushCount").setValue(pushId+1);
                    }

                    @Override
                    public void onChildChanged(@NonNull DataSnapshot snapshot, @Nullable String previousChildName) {

                    }

                    @Override
                    public void onChildRemoved(@NonNull DataSnapshot snapshot) {

                    }

                    @Override
                    public void onChildMoved(@NonNull DataSnapshot snapshot, @Nullable String previousChildName) {

                    }

                    @Override
                    public void onCancelled(@NonNull DatabaseError error) {

                    }
                });
        pushCount += 1;
        System.out.println("SAVE COMPLETE");
        readPush();
    }

    public void readPush() {
        mDatabase.child("push").child(String.valueOf(pushCount-1)).addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                System.out.println("ADDCHILD");
                // Get Post object and use the values to update the UI
                if (snapshot.getValue(Push.class) != null) {
                    Push push = snapshot.getValue(Push.class);
                    pushList.add(push);
                    System.out.println("push.getContent() = " + push.getContent());

                    System.out.println("pushList = " + pushList);
                    ((MainActivity)MainActivity.mContext).drawChart(pushList, pushCount);
                } else {
                    System.out.println("Non-Data.");
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {
                // Getting Post failed, log a message
                Log.w("FireBaseData", "loadPost:onCancelled", databaseError.toException());
            }
        });
    }
}
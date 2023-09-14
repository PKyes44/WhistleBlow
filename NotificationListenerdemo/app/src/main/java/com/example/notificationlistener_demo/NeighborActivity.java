package com.example.notificationlistener_demo;

import android.annotation.SuppressLint;
import android.app.PendingIntent;
import android.content.Intent;
import android.os.Bundle;
import android.telephony.SmsManager;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import org.w3c.dom.Text;

import java.util.ArrayList;

public class NeighborActivity extends AppCompatActivity  {
    private ArrayList<Neighbor> mArrayList;
    private NeighborAdapter mAdapter;
    private int count = -1;
    private int neighborCount;
    private DatabaseReference mDatabase;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.neighbor_setting);

        RecyclerView mRecyclerView = findViewById(R.id.recyclerView);
        LinearLayoutManager mLinearLayoutManager = new LinearLayoutManager(this);
        mRecyclerView.setLayoutManager(mLinearLayoutManager);


        mArrayList = new ArrayList<>();
        mDatabase = FirebaseDatabase.getInstance().getReference();

        mAdapter = new NeighborAdapter(mArrayList);
        mRecyclerView.setAdapter(mAdapter);


        DividerItemDecoration dividerItemDecoration = new DividerItemDecoration(mRecyclerView.getContext(),
                mLinearLayoutManager.getOrientation());
        mRecyclerView.addItemDecoration(dividerItemDecoration);



        @SuppressLint({"MissingInflatedId", "LocalSuppress"}) Button buttonInsert = findViewById(R.id.neighbor_button);
        buttonInsert.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                count++;
                TextView phoneNum = findViewById(R.id.phoneNumEdit);

                Neighbor data = new Neighbor(String.valueOf(count), String.valueOf(phoneNum.getText()));

                //mArrayList.add(0, dict); //RecyclerView의 첫 줄에 삽입
                mArrayList.add(data); // RecyclerView의 마지막 줄에 삽입

                mAdapter.notifyDataSetChanged();
                writeNeighbor(String.valueOf(phoneNum.getText()));
            }
        });

        @SuppressLint({"MissingInflatedId", "LocalSuppress"}) ImageView goMainBtn = findViewById(R.id.goMain_btn);
        goMainBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(NeighborActivity.this, MainActivity.class);
                startActivity(intent);
                finish();
            }
        });

    }


    public void writeNeighbor(String phone) {
        mDatabase.child("neighbor").child(String.valueOf(neighborCount++)).setValue(phone)
                .addOnSuccessListener(new OnSuccessListener<Void>() {
                    @Override
                    public void onSuccess(Void unused) {
                        SmsSend(phone, "주변인으로 등록되셨습니다");
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                    }
                });
    }

    public void SmsSend(String strPhoneNumber, String strMsg){

        PendingIntent sendIntent = PendingIntent.getBroadcast(this, 0, new Intent("SMS_SENT"), PendingIntent.FLAG_IMMUTABLE);
        PendingIntent deliveredIntent = PendingIntent.getBroadcast(this, 0, new Intent("SMS_DELIVERED"), PendingIntent.FLAG_IMMUTABLE);


        SmsManager smsManager = SmsManager.getDefault();
        try {
            smsManager.sendTextMessage(strPhoneNumber, null, strMsg, sendIntent, deliveredIntent);
        } catch (Exception ex) {
            ex.printStackTrace();
            Toast.makeText(getBaseContext(), ex.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
}

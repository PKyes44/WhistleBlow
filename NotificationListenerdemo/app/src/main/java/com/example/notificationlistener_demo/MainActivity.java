package com.example.notificationlistener_demo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;
import androidx.core.content.ContextCompat;

import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.components.Description;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.data.PieEntry;
import com.github.mikephil.charting.utils.ColorTemplate;
import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    PieChart pieChart;
    private DatabaseReference mDatabase;

    public static Context mContext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getSupportActionBar().hide();
        setContentView(R.layout.activity_main);
        mContext = this;

        mDatabase = FirebaseDatabase.getInstance().getReference();

        @SuppressLint({"MissingInflatedId", "LocalSuppress"}) AppCompatButton setNeighbor = findViewById(R.id.goSetNeighbor_btn);
        setNeighbor.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, NeighborActivity.class);
                startActivity(intent);
                finish();
            }
        });
    }

    public void drawChart(ArrayList<Push> pushList, int pushCount) {
        System.out.println("DRAWCHART");

        pieChart = findViewById(R.id.piechart);
        View detectView = findViewById(R.id.detectView);
        TextView detectTextView = findViewById(R.id.detectText);

        pieChart.setUsePercentValues(true);
        pieChart.getDescription().setEnabled(false);
        pieChart.setExtraOffsets(5, 10, 5, 5);

        pieChart.setDragDecelerationFrictionCoef(0.95f);

        pieChart.setDrawHoleEnabled(false);
        pieChart.setHoleColor(Color.WHITE);
        pieChart.setTransparentCircleRadius(61f);

        float maliciousCount;
        float unMaliciousCount;

        if (pushList.isEmpty()) {
            maliciousCount = 0f;
            unMaliciousCount = 100f;
        } else {
            System.out.println("pushCount = " + pushCount);
            maliciousCount = (pushList.size()) / pushCount * 100;
            unMaliciousCount = (pushCount-(pushList.size())) / pushCount * 100;
        }

        System.out.println("maliciousCount = " + maliciousCount);
        System.out.println("unMaliciousCount = " + unMaliciousCount);
        PieEntry malicious = new PieEntry(maliciousCount, "악의적인 채팅");
        PieEntry unMalicious = new PieEntry(unMaliciousCount, "악의적이지 않은 채팅");

        if (maliciousCount <= 80) {
            detectView.setBackgroundColor(Color.parseColor("#6CE461"));
            detectTextView.setText("당신은 안전한 상태입니다");
        } else {
            detectView.setBackgroundColor(Color.parseColor("#FF5942"));
            detectTextView.setText("당신은 위험한 상태입니다");
        }

        ArrayList<PieEntry> yValues = new ArrayList<PieEntry>();
        yValues.add(malicious);
        yValues.add(unMalicious);

        Description description = new Description();
        description.setText(""); //라벨
        description.setTextSize(15);
        pieChart.setDescription(description);

        PieDataSet dataSet = new PieDataSet(yValues,"");
        dataSet.setSliceSpace(3f);
        dataSet.setSelectionShift(5f);
        dataSet.setColors(ColorTemplate.JOYFUL_COLORS);

        PieData data = new PieData((dataSet));
        data.setValueTextSize(10f);
        data.setValueTextColor(Color.YELLOW);

        pieChart.setCenterText(pushList.size() + "개"); // 중앙에 표시할 텍스트 설정
        pieChart.setCenterTextSize(16f); // 중앙 텍스트의 크기 설정
        pieChart.setCenterTextColor(Color.BLACK); // 중앙 텍스트의 색상 설정

        pieChart.setData(data);
        pieChart.invalidate(); // 그래프 업데이트
    }
}
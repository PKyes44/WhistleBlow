package com.example.notificationlistener_demo;

import android.util.TypedValue;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class NeighborAdapter extends RecyclerView.Adapter<NeighborAdapter.CustomViewHolder> {
    private ArrayList<Neighbor> mList;

    public class CustomViewHolder extends RecyclerView.ViewHolder {
        protected TextView id;
        protected TextView phone;


        public CustomViewHolder(View view) {
            super(view);
            this.id = (TextView) view.findViewById(R.id.id_listitem);
            this.phone = (TextView) view.findViewById(R.id.phone_listItem);
        }
    }


    public NeighborAdapter(ArrayList<Neighbor> list) {
        this.mList = list;
    }

    @Override
    public CustomViewHolder onCreateViewHolder(ViewGroup viewGroup, int viewType) {

        View view = LayoutInflater.from(viewGroup.getContext())
                .inflate(R.layout.neightbor_list, viewGroup, false);

        CustomViewHolder viewHolder = new CustomViewHolder(view);

        return viewHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull CustomViewHolder viewHolder, int position) {

        viewHolder.id.setTextSize(TypedValue.COMPLEX_UNIT_SP, 25);
        viewHolder.phone.setTextSize(TypedValue.COMPLEX_UNIT_SP, 25);

        viewHolder.id.setGravity(Gravity.CENTER);
        viewHolder.phone.setGravity(Gravity.LEFT);

        viewHolder.id.setText(mList.get(position).getId());
        viewHolder.phone.setText(mList.get(position).getPhone());
    }

    @Override
    public int getItemCount() {
        return (null != mList ? mList.size() : 0);
    }

}

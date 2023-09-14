package com.example.notificationlistener_demo;

import com.google.firebase.database.IgnoreExtraProperties;

@IgnoreExtraProperties
public class Push {
    public String sender;
    public String content;
    public String sendTime;

    public Push() {
        // Default constructor required for calls to DataSnapshot.getValue(User.class)
    }

    public String getSender() {
        return sender;
    }
    public String getContent() {
        return content;
    }
    public String getSendTime() {
        return sendTime;
    }

    public void setSender(String sender) {
        this.sender = sender;
    }
    public void setContent(String content) {
        this.content = content;
    }
    public void setSendTime(String SendTime) {
        this.sendTime = sendTime;
    }

    public Push(String sender, String content, String sendTime) {
        this.sender = sender;
        this.content = content;
        this.sendTime = sendTime;
    }
}

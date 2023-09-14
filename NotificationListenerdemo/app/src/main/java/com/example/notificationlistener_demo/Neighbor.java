package com.example.notificationlistener_demo;

public class Neighbor {
    private String id;
    private String phone;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }
    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }

    public Neighbor(String id, String phone) {
        this.id = id;
        this.phone = phone;
    }
}

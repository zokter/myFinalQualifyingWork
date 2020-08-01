const express = require("express");
const app = express();

app.use("/static", express.static("static"));

app.use(function(req, res, next){
    console.log(`${new Date()} - ${req.method} request for ${req.url}`);
    next();//pass control to the next handler 
});

app.get("/" , function(req, res){
    res.sendFile(__dirname + "/static/main.html");
})

app.listen(3000, function(){
    console.log("on air");
});

app.get("/main.html" , function(req, res){
    res.sendFile(__dirname + "/static/main.html");
})
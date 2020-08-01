var modelPath = "http://localhost:3000/static/TFJSModel/myOwnModel/model.json";

$("#image-selector").change(function(){
    let reader = new FileReader();
    reader.onload = function(){
        let dataURL = reader.result;
        $("#selected-image").attr("src",dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
});

var model = 0;
(async function(){
    model = await tf.loadLayersModel(modelPath);
    $(".progress-bar").hide();
})();

$("#predict-button").click(async function(){
    let image = $("#selected-image").get(0);
    let tensor = tf.browser.fromPixels(image);
        tensor = tf.image.resizeNearestNeighbor(tensor, [224,224])
        .toFloat()
        .div(255)
        .expandDims();
    let predictions = await model.predict(tensor).data();
    let top5 = Array.from(predictions)
        .map(function(p, i){
            return {
                probability: p,
                className: MYOWN_CLASSES[i]
            };
        }).sort(function (a, b){
            return b.probability - a.probability;
        }).slice(0, 5);
    $("#prediction-list").empty();
    top5.forEach(function (p){
        $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`)
    })

    objectPresentation(top5[0]);
})

function objectPresentation(predictedObject){
    var canvas = document.getElementById("canvas");
    var renderer = new THREE.WebGLRenderer({canvas: canvas});
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
   
    camera.position.set(0, 0, 2);
    renderer.setClearColor(0x333333);

    
    var keyLight = new THREE.DirectionalLight(new THREE.Color("hsl(30, 100%, 75%)"), 1.0);
    keyLight.position.set(-100, 0, 100);

    var fillLight = new THREE.DirectionalLight(new THREE.Color("hsl(240, 140%, 75%)"), 0.75);
    fillLight.position.set(100, 0, 100);

    var backLight = new THREE.DirectionalLight(0xffffff, 1.0);
    backLight.position.set(100, 0, 100).normalize();

    scene.add(keyLight);
    scene.add(fillLight);
    scene.add(backLight);

    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.campingFactor = 0.25;
    controls.enableZoom = true;

    if(predictedObject.className == "bottle"){
        var objLoader = new THREE.OBJLoader();
        objLoader.setPath("/static/3D/bottle/");
        objLoader.load("Bottle_blend.obj", function(object){
            object.position.y -= 0;
            scene.add(object);
        });
    }

    if(predictedObject.className == "cup"){
        var objLoader = new THREE.OBJLoader();
        objLoader.setPath("/static/3D/cup/");
        objLoader.load("cupa.obj.obj", function(object){
            object.position.y -= 0;
            scene.add(object);
        });
    }

    if(predictedObject.className == "kettle"){
        var objLoader = new THREE.OBJLoader();
        objLoader.setPath("/static/3D/teapot/");
        objLoader.load("Teapot.obj", function(object){
            object.position.y -= 2;
            scene.add(object);
        });
    }

    var animate = function(){
        requestAnimationFrame(animate);

        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}


  // for MobileNet
        // let offset = tf.scalar(127.5);
        // tensor = tensor.sub(offset)
        // .div(offset)
        // .expandDims();


    // let predictions = await model.predict(tensor).data();
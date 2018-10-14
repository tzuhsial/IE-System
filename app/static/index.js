var sampleUrl = location.origin + "/sample";
var openUrl = location.origin + "/open";
var stepUrl = location.origin + "/step";
var resetUrl = location.origin + "/reset";

// States
// gesture_click
var xcor = -1;
var ycor = -1;

// Bounding box
var editor = null;
var bb_cors = null;

var previewFile = function () {
    var preview = document.querySelector('#image');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.addEventListener("load", function () {
        preview.src = reader.result;
        var b64_img_str = preview.src.split(',')[1];
        submitImage(b64_img_str);
    }, false);

    if (file) {
        reader.readAsDataURL(file);
    }
}


var submitImage = function (b64_img_str) {
    data = {}
    data["b64_img_str"] = b64_img_str;

    toggleLoading(true);
    $.post(openUrl, data, function (response) {
        console.log(response);
        $("#image").attr("src", "data:image/png;base64," + response["b64_img_str"]);
    }).always(function () {
        toggleLoading(false);
    })
}



var submitRequest = function (user_utterance) {

    data = {}
    data['user_utterance'] = user_utterance;

    // Add Gestures slots
    if ($("#gesture_click_mode").is(":checked")) {
        if (xcor >= 0 && ycor >= 0) {
            console.log("adding gesture_click");
            var gesture_click = {};
            gesture_click["x"] = xcor;
            gesture_click["y"] = ycor;
            data["gesture_click"] = JSON.stringify(gesture_click);
        }
        // Uncheck
        $("#gesture_click_mode").prop("checked", false);
        $("#marker").hide();
        xcor = -1;
        ycor = -1;
    }

    if ($("#object_mask_str_mode").is(":checked")) {
        if (bb_cors != null) {
            data["object_mask_str"] = JSON.stringify(bb_cors);
            editor.clear_all();
        }
        $("#object_mask_str_mode").prop("checked", false);
        $("#bbox_annotator").empty();
        bb_cors = null;

        $("#image-container").show();
    }

    console.log('submit', data);

    toggleLoading(true);
    $.post(stepUrl, data, function (response) {
        console.log(response);
        $("#image").attr("src", "data:image/png;base64," + response["b64_img_str"]);
        $("#system_utterance").text(response["system_utterance"])
    }).always(function () {
        toggleLoading(false);
    })
}

var sampleGoal = function () {
    //console.log("Randomly sampling a goal...");
    toggleLoading(true);
    $.post(sampleUrl, {}, function (response) {
        console.log(response);
        var b64_img_str = response["b64_img_str"];
        $("#image").attr("src", "data:image/png;base64," + b64_img_str);

        var goal = response["goal"];
        var object_mask_img_str = goal["object_mask_img_str"];
        //console.log("goal", goal);
        $("#object_mask").attr("src", "data:image/png;base64," + object_mask_img_str);

        var goal_text = "";
        goal_text += "intent=" + goal["intent"];
        goal_text += ", object=" + goal["object"];
        goal_text += ", attribute=" + goal["attribute"];
        goal_text += ", adjust_value=" + goal["adjust_value"];
        $("#goal-text").text(goal_text);

        $("#system_utterance").text(response["system_utterance"]);
    }).always(function () {
        toggleLoading(false);
    })
}

var toggleLoading = function (show) {
    if (show) {
        $("#loading-container").show();
    } else {
        $("#loading-container").hide();
    }
}

$(document).ready(function () {

    $("#reset-button").on('click', function (e) {
        e.preventDefault();

        toggleLoading(true);
        $.post(resetUrl, {}, function (response) {
            console.log("reset");
        }).always(function () {
            toggleLoading(false);
        });

    });

    $("#sample-button").on('click', function (e) {
        e.preventDefault();
        sampleGoal();
    });

    $("#submit").on('click', function (e) {
        e.preventDefault();

        var user_utterance = $("#user_utterance").val();
        console.log(user_utterance);

        submitRequest(user_utterance);

        $("#user_utterance").val("");
    });

    $("#image").on('click', function (event) {
        var offsetLeft = this.offsetLeft;
        var offsetTop = this.offsetTop;

        var i = $("#image").get(0);

        var naturalWidth = i.naturalWidth;
        var naturalHeight = i.naturalHeight;
        var clientWidth = i.clientWidth;
        var clientHeight = i.clientHeight;

        if ($("#gesture_click_mode").is(":checked")) {
            $('#marker').css('left', event.pageX).css('top', event.pageY).show();
            client_ycor = (event.pageX - offsetLeft);
            client_xcor = (event.pageY - offsetTop);
            console.log("Client X Coordinate: " + client_xcor + ", Y Coordinate: " + client_ycor);
            xcor = Math.round(client_xcor * naturalHeight / clientHeight);
            ycor = Math.round(client_ycor * naturalWidth / clientWidth);
            console.log("X Coordinate: " + xcor + ", Y Coordinate: " + ycor);
        } else {
            xcor = -1;
            ycor = -1;
        }
    });

    $("#gesture_click_mode").change(function () {
        if (this.checked) {
            console.log("gesture_click is checked!")
        } else {
            $("#marker").hide();
            xcor = -1;
            ycor = -1;
        }
    });

    $("#object_mask_str_mode").change(function () {
        if (this.checked) {
            $("#image-container").hide();

            console.log("object_mask_str is checked!")
            editor = new BBoxAnnotator({
                url: $("#image").attr("src"),
                multiple: false,
                onchange: function (annotation) {
                    console.log(annotation[0]);
                    bb_cors = annotation[0];
                }
            });
        } else {
            editor.clear_all();
            $("#bbox_annotator").empty();
            $("#image-container").show();
        }
    });

    $("#object_mask-button").on('click', function (event) {
        $("#goal-container").toggle();
    });
    sampleGoal();
});
var sampleUrl = location.origin + "/sample";
var stepUrl = location.origin + "/step";
var resetUrl = location.origin + "/reset";
var resultUrl = location.origin + "/result";

// SessionID
var d = new Date();
var n = d.getTime();
var session_id = n.toString();
var turn_count = 1;

// States
// gesture_click
var xcor = -1;
var ycor = -1;

// Bounding box
var editor = null;
var bb_cors = null;

var updateTurnCount = function (inc = 1) {
    turn_count += inc;
    $("#turn-count").text("Turn Count: " + turn_count);
}

var submitRequest = function (user_utterance) {

    var data = {}
    data["session_id"] = session_id;
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
        $("#image").attr("src", "data:image/png;base64," + response["b64_img_str"]);

        var sys_utt = response["system_utterance"];
        var last_execute_result = response["last_execute_result"];
        console.log(response)

        updateTurnCount();

        if (sys_utt.includes("Execute")) {
            sys_utt += " Execution result " + last_execute_result.toString() + ".";
        }

        if (turn_count >= 10 || (sys_utt.includes("Execute") & last_execute_result)) {

            sys_utt += " You may now click on End Dialogue ";
            $("#end-modal").modal('show');
        }
        $("#system_utterance").text(sys_utt);

    }).always(function () {
        toggleLoading(false);
    })
}

var sampleGoal = function (goal_idx = -1) {
    //console.log("Randomly sampling a goal...");
    var data = {
        "session_id": session_id,
        "goal_idx": goal_idx
    }
    toggleLoading(true);
    $.post(sampleUrl, data, function (response) {
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
        updateTurnCount(-turn_count + 1);
    }).always(function () {
        toggleLoading(false);
    })
}

var submitReset = function () {
    var data = {
        "session_id": session_id
    }
    toggleLoading(true);
    $.post(resetUrl, data, function (response) {
        console.log("reset");
        $("#image").attr("src", "data:image/png;base64," + response["b64_img_str"]);
        $("#system_utterance").text(response["system_utterance"])
        updateTurnCount(-turn_count);
    }).always(function () {
        toggleLoading(false);
    });
}

var submitResult = function (result) {
    var data = {
        "session_id": session_id,
        "result": result
    }
    console.log("submit", data);
    toggleLoading(true);
    $.post(resultUrl, data, function (response) {
        console.log(response);
        $("#system_utterance").text(response["system_utterance"])
        $("#end-modal").modal("show");
    }).always(function () {
        toggleLoading(false);
    });
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
        submitReset();
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
        if (xcor >= 0 && ycor >= 0) {
            var container_height = $("#goal-container").height();
            var marker_pos = $("#marker").offset();
            if ($("#goal-container").is(":hidden")) {

                $('#marker').css('top', marker_pos["top"] + container_height).show();
            } else {
                $('#marker').css('top', marker_pos["top"] - container_height).show();
            }
        }

        $("#goal-container").toggle();
    });

    $("select").on('change', function (e) {
        var goal_idx = parseInt(this.value);
        sampleGoal(goal_idx);
    })


    $("#success-button").on('click', function (event) {
        submitResult("success");
    });
    $("#failure-button").on('click', function (event) {
        submitResult("failure");
    });
    //sampleGoal();

    $("#end-button").on("click", function () {
        $("#end-modal").modal("show");
    })
});
var initUrl = location.origin + "/init";
var stepUrl = location.origin + "/step";
var resetUrl = location.origin + "/reset";
var resultUrl = location.origin + "/result";

// SessionID
var d = new Date();
var session_id = -1;
var turn_count = 0;
var max_turn = 10;


var updateTurnCount = function (inc = 1) {
    turn_count += inc;
    $("#turn-count").text("Turn: " + turn_count);
}

var submitInit = function () {
    var data = {}
    toggleLoading(true);
    $.post(initUrl, data, function (response) {
        $("#image").attr("src", "data:image/png;base64," + response["b64_img_str"]);
        var sys_utt = response["system_utterance"];
        $("#system_utterance").text(sys_utt);
        updateTurnCount()
    }).fail(function() {
        console.error("Failed to initialize!")
    }).always(function () {
        toggleLoading(false);
    });
}

var submitRequest = function (user_utterance) {

    var data = {}
    data["session_id"] = session_id;
    data['user_utterance'] = user_utterance;

    console.log('submit', data);

    toggleLoading(true);
    $.post(stepUrl, data, function (response) {
        $("#image").attr("src", "data:image/png;base64," + response["b64_img_str"]);

        var sys_utt = response["system_utterance"];

        updateTurnCount();

        $("#system_utterance").text(sys_utt);

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

var update_slider = function (attribute, value) {
    $("#input_" + attribute).val(value);
    $("#output_" + attribute).text(value);
}

var update_object = function (object) {
    $("#object").text(object);
}


// Disables user input
var prepare_survey = function () {
    $("#end-button").prop('disabled', false);
    $("#survey-button").prop('disabled', false);
}

// Disable inputs
var disable_inputs = function() {
    // Disable input
    $("#user_utterance").prop('disabled', true);
    // Disable survey
    $("#end-button").prop('disabled', true);
    $("#survey-button").prop('disabled', true);
    // Pop a sorry modal
    $("#sorry-modal").modal('show');
}


var submitSurvey = function() {
      
    $("#survey-code").text(session_id);
    $("#survey-button").prop('disabled', true);
}

var get_survey_code = function() {
    $("#survey-button").text(session_id);
}

$(document).ready(function () {
    $("#submit").on('click', function (e) {
        e.preventDefault();

        var user_utterance = $("#user_utterance").val();
        console.log(user_utterance);

        submitRequest(user_utterance);

        $("#user_utterance").val("");
    });


    $("#end-button").on("click", function () {
        $("#end-modal").modal("show");
    })

    $("#survey-button").on('click', function() {
        submitSurvey();
    });

    var attributes = ["brightness", "contrast", "hue", "saturation", "lightness"];
    for (var i = 0; i < 5; i++) {
        var attribute = attributes[i];
        // Initialize slider
        $("#input_" + attribute).slider();
    }

    setTimeout(function () {
        submitInit();
    }, 1000);
});
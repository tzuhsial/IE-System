var initUrl = location.origin + "/init";
var stepUrl = location.origin + "/step";
var surveyUrl = location.origin + "/survey";

// SessionID
var d = new Date();
var session_id = -1;
var turn_count = 0;
var edit_count = 0;
var min_edit = 2;
var min_turn = 10;
var end_turn = 30;

var updateTurnCount = function (inc = 1) {
    turn_count += inc;
    $("#turn-count").text("Turn: " + turn_count + "/" + min_turn);

    // Users manage to do nothing for 20 turns.
    if (turn_count >= min_turn && edit_count >= min_edit) {
        $("#end-button").prop('disabled', false);
    }

    if (turn_count >= end_turn) {
        $("#end-button").prop('disabled', false);
    }
}

var updateEditCount = function (inc = 1) {
    edit_count += inc;
    $("#edit-count").text("Edits: " + edit_count + "/" + min_edit);
}

var submitInit = function () {
    var data = {
        "session_id": session_id
    }
    toggleLoading(true);
    $.post(initUrl, data, function (response) {
        $("#image").attr("src", "data:image/png;base64," + response["b64_img_str"]);
        var sys_utt = response["system_utterance"];
        $("#system_utterance").text(sys_utt);
        // Start from 0
        updateTurnCount(0);
        updateEditCount(0);
    }).fail(function () {
        console.error("Failed to initialize!")
        server_error();
    }).always(function () {
        toggleLoading(false);
    });
}

var submitStep = function (user_utterance) {

    var data = {}
    data["session_id"] = session_id;
    data['user_utterance'] = user_utterance;

    //console.log('submit', data);

    toggleLoading(true);
    $.post(stepUrl, data, function (response) {
        $("#image").attr("src", "data:image/png;base64," + response["b64_img_str"]);

        var sys_utt = response["system_utterance"];

        //console.log(response);
        // Show real users what is really happening.

        var object = response['object'];
        //console.log('object', object);
        if (object != null) {
            update_object(object);
        } else {
            update_object("(Not detected)");
        }

        var sys_act = response['system_act']
        var da = sys_act['dialogue_act']['value']

        // Update slider if executed
        if (da == "execute") {
            var slots = sys_act['slots'];
            var attribute = null;
            var adjust_value = null;

            for (var i = 0; i < slots.length; i++) {
                var slot = slots[i];
                if (slot['slot'] == "attribute") {
                    attribute = slot['value'];
                } else if (slot['slot'] == "adjust_value") {
                    adjust_value = parseInt(slot['value']);
                }
            }

            // Reset sliders
            var attributes = ["brightness", "contrast", "hue", "saturation", "lightness"];
            for (var i = 0; i < attributes.length; i++) {
                update_slider(attributes[i], 0);
            }

            update_slider(attribute, adjust_value);

            // Activate end
            updateEditCount();

            // Need to satisfy both conditions:
            // Edits 2
            // Turns 10
            if (edit_count >= min_edit && turn_count >= min_turn) {
                $("#end-button").prop('disabled', false);
            }
        }

        updateTurnCount();

        $("#system_utterance").text(sys_utt);
        $("#system_utterance2").prop("value", sys_utt);

    }).fail(function () {
        server_error();
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

// Disable inputs
var server_error = function () {
    // Disable input
    $("#user_utterance").prop('disabled', true);
    // Disable survey
    $("#end-button").prop('disabled', true);
    $("#survey-button").prop('disabled', true);
    // Pop a sorry modal
    $("#sorry-modal").modal('show');
}


var submitSurvey = function () {

    var data = {}
    data["session_id"] = session_id;

    var survey = get_survey_data()
    if (survey == null) {
        alert("Please fill in all the answers before submitting!");
        return;
    }
    data["survey"] = JSON.stringify(survey);

    toggleLoading(true);
    $.post(surveyUrl, data, function (response) {
        console.log("Successfully recorded!")
        // Show survey code to user
        $("#survey-code").text(session_id);
        $("#system_utterance").text("You may now close the window and submit your survey code.");
        // Disable everything else
        $("#user_utterance").prop('disabled', true);
        $("#survey-button").prop('disabled', true);
    }).fail(function () {
        console.error("Failed to record!");
        $("#end-modal").modal("hide");
        server_error();
    }).always(function () {
        toggleLoading(false);
    })


}

var get_survey_data = function () {
    var survey = {}

    // Get Multiple Choice Answers 
    var multiple_choices = ["performance", "nlu-object", "cv", "nlu-attribute-value"];

    for (var i = 0; i < multiple_choices.length; i++) {
        var mp = multiple_choices[i];
        var answer = $("#" + mp + " input:radio:checked").val();
        if (answer == null) {
            return null;
        }
        survey[mp] = answer;
    }

    // Get feedback answers
    var feedback_types = ["like", "dislike", "suggestion"];
    for (var i = 0; i < feedback_types.length; i++) {
        var ft = feedback_types[i];
        var answer = $("#input-" + ft).val();
        if (answer.trim() == "") {
            return null;
        }
        survey[ft] = answer;
    }

    return survey
}

var get_survey_code = function () {
    $("#survey-button").text(session_id);
}

$(document).ready(function () {

    $("#submit").on('click', function (e) {
        e.preventDefault();

        var user_utterance = $("#user_utterance").val();
        if (user_utterance.trim() != "") {
            submitStep(user_utterance);
        }
        $("#user_utterance").val("");
    });

    $("#user_utterance").on('keypress', function (e) {
        if (e.keyCode == 13) {
            $("#submit").click();
        }
    });


    $("#end-button").on("click", function () {
        $("#end-modal").modal("show");
    })

    $("#survey-button").on('click', function () {
        submitSurvey();
    });

    // Initialize slider
    var attributes = ["brightness", "contrast", "hue", "saturation", "lightness"];
    for (var i = 0; i < 5; i++) {
        var attribute = attributes[i];
        $("#input_" + attribute).slider();
        $("#input_" + attribute).on("slide", function (event, ui) {
            return false;
        });
    }

    // Disable end dialogue button
    $("#end-button").prop('disabled', true);

    $("#instruction-button").on("click", function () {
        $("#instruction-modal").modal('show');
    });


    setTimeout(function () {
        submitInit();
    }, 1000);

    $("#instruction-modal").modal('show');
    $("#user_utterance").focus();
});
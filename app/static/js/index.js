var stepUrl = location.origin + "/step";
var resetUrl = location.origin + "/reset";
var resultUrl = location.origin + "/result";

// SessionID
var d = new Date();
var n = d.getTime();
var session_id = n.toString();
var turn_count = 1;
var max_turn = 10;


var updateTurnCount = function (inc = 1) {
    turn_count += inc;
    $("#turn-count").text("Turn Count: " + turn_count);
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
        var last_execute_result = response["last_execute_result"];

        updateTurnCount();

        if (turn_count >= max_turn || (sys_utt.includes("Adjust") & last_execute_result)) {
            sys_utt += " You may now click on End Dialogue ";
            //$("#end-modal").modal('show');
        }
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

$(document).ready(function () {


    $("#submit").on('click', function (e) {
        e.preventDefault();

        var user_utterance = $("#user_utterance").val();
        console.log(user_utterance);

        submitRequest(user_utterance);

        $("#user_utterance").val("");
    });


    $("#success-button").on('click', function (event) {
        submitResult("success");
    });
    $("#failure-button").on('click', function (event) {
        submitResult("failure");
    });

    $("#end-button").on("click", function () {
        $("#end-modal").modal("show");
    })

    updateTurnCount(0);
});
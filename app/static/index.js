/**
 * Javascript file for Simple Photoshop Demo
 */
var ierUrl = location.origin + "/ier";

var previewFile = function () {
    var preview = document.querySelector('img');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.addEventListener("load", function () {
        preview.src = reader.result;
        var b64_img_str = preview.src.split(',')[1];

        var observation = {
            'user_acts': [
                {
                    'dialogue_act': 'load',
                    'slots': [
                        { 'slot': 'action_type', 'value': 'load' },
                        { 'slot': 'b64_img_str', 'value': b64_img_str }
                    ]
                }
            ]
        }
        submitRequest(observation);
    }, false);

    if (file) {
        reader.readAsDataURL(file);
    }
}

var getObservation = function () {
    // Parse slot values here
    var dialogue_act = $("#dialogueact").val();

    var slots = [];
    for (var i = 1; i <= 4; ++i) {
        var slot_id = "#slot" + i.toString();
        var value_id = "#value" + i.toString();
        var conf_id = "#conf" + i.toString();
        if ($(slot_id).val().trim() != "" && $(value_id).val().trim() != "" && $(conf_id).val().trim() != "") {
            var slot_dict = {
                'slot': $(slot_id).val(),
                'value': $(value_id).val(),
                'conf': parseFloat($(conf_id).val())
            };
            slots.push(slot_dict)
        }
    }

    var observation = {
        'user_acts': [
            {
                'dialogue_act': dialogue_act,
                'slots': slots
            }
        ]
    };
}

var submitRequest = function (observation) {

    if (observation['user_acts']['slots'].length == 0) return;

    var data = {
        'observation': JSON.stringify(observation)
    };

    console.log('submit', data);


    toggleLoading(true);
    $.post(ierUrl, data, function (response) {
        loadImage(response['b64_img_str']);
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

var loadImage = function (b64_img_str) {
    var display_string = "data:image/png;base64," + b64_img_str;
    $("#image").attr('src', display_string);
}


$(document).ready(function () {

    $("#submit").click(function () {
        observation = getObservation();
        submitRequest(observation);
    });
});
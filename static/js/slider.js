// Tasks and Data
const tasks = ["representation", "layerwise", "noise"];

const data = {
    "representation": {
        "use_20": false,
        "use_two_sided": false,
        "data" : [
            {
                title: [
                    "<strong>PCA visualization of queries, keys, values, and intermediate features.</strong> Queries and keys preserve structural cues for geometric matching, while intermediate features cluster semantically, integrating appearance details from values, potentially weakening structural cues for correspondence.",
                    "<strong>Evolution of attention scores across timesteps.</strong> As noise decreases, text-frame and self-frame attention diminish, reducing reliance on textual guidance, while cross-frame attention strengthens and converges, improving temporal coherence.",
                    "<strong>Impact of positional bias on matching accuracy and affinity score.</strong> (a) Top-20 timesteps and layers where affinity scores are high but matching accuracies are low. (b) Matching cost visualization shows points in the first frame attending to their initial locations rather than matched counterparts, highlighting positional bias. (c) PCA of queries and keys in the top-$1$ layer and timestep from (a), showing dominant positional cues.",
                    // "<strong>Mismatch between matching accuracy and affinity score due ot persistent text attention.</strong> (a) Top-20 timesteps and layers where matching accuracy is high but affinity score is low. (b) Text-frame attention remains high, suppressing affinity scores and weakening cross-frame interactions.",
                ],
                images: [
                    "static/images/pca.png",
                    "static/images/analysis_layer.png",
                    "static/images/self_attend.png",
                    
                ]
            },
          
        ]
    },
    "layerwise": {
        "use_20": false,
        "use_two_sided": false,
        "data" : [
            {
                title: [
                    "<strong>PCA visualization of queries, keys, values, and intermediate features.</strong> Queries and keys preserve structural cues for geometric matching, while intermediate features cluster semantically, integrating appearance details from values, potentially weakening structural cues for correspondence.",
                    "<strong>Evolution of attention scores across timesteps.</strong> As noise decreases, text-frame and self-frame attention diminish, reducing reliance on textual guidance, while cross-frame attention strengthens and converges, improving temporal coherence.",
                    "<strong>Impact of positional bias on matching accuracy and affinity score.</strong> (a) Top-20 timesteps and layers where affinity scores are high but matching accuracies are low. (b) Matching cost visualization shows points in the first frame attending to their initial locations rather than matched counterparts, highlighting positional bias. (c) PCA of queries and keys in the top-$1$ layer and timestep from (a), showing dominant positional cues.",
                    // "<strong>Mismatch between matching accuracy and affinity score due ot persistent text attention.</strong> (a) Top-20 timesteps and layers where matching accuracy is high but affinity score is low. (b) Text-frame attention remains high, suppressing affinity scores and weakening cross-frame interactions.",
                ],
                images: [
                    "static/images/pca.png",
                    "static/images/analysis_layer.png",
                    "static/images/self_attend.png",
                ]
            },
          
        ]
    },
    "noise": {
        "use_20": false,
        "use_two_sided": false,
        "data" : [
            {
                title: [
                    "<strong>PCA visualization of queries, keys, values, and intermediate features.</strong> Queries and keys preserve structural cues for geometric matching, while intermediate features cluster semantically, integrating appearance details from values, potentially weakening structural cues for correspondence.",
                    "<strong>Evolution of attention scores across timesteps.</strong> As noise decreases, text-frame and self-frame attention diminish, reducing reliance on textual guidance, while cross-frame attention strengthens and converges, improving temporal coherence.",
                    "<strong>Impact of positional bias on matching accuracy and affinity score.</strong> (a) Top-20 timesteps and layers where affinity scores are high but matching accuracies are low. (b) Matching cost visualization shows points in the first frame attending to their initial locations rather than matched counterparts, highlighting positional bias. (c) PCA of queries and keys in the top-$1$ layer and timestep from (a), showing dominant positional cues.",
                    // "<strong>Mismatch between matching accuracy and affinity score due ot persistent text attention.</strong> (a) Top-20 timesteps and layers where matching accuracy is high but affinity score is low. (b) Text-frame attention remains high, suppressing affinity scores and weakening cross-frame interactions.",
                ],
                images: [
                    "static/images/pca.png",
                    "static/images/analysis_layer.png",
                    "static/images/self_attend.png",
                ]
            },
          
        ]
    },

};

let slideIndex = 1;
let currentSceneIndex = {};
let twenty_inited = {};
for (let i = 0; i < tasks.length; i++) {
    currentSceneIndex[tasks[i]] = 0;
    twenty_inited[tasks[i]] = false;
}

const garmentSelected = {
    A: "1",
    B: "1",
    C: "1"
};

// Mapping combinations to images
const tryonSceneIndex = {
    "1-1-1": "0",
    "1-1-2": "1",
    "1-2-1": "2",
    "1-2-2": "3",
    "2-1-1": "4",
    "2-1-2": "5",
    "2-2-1": "6",
    "2-2-2": "7"
};

const sceneIndexToInput = [
    "1-1-1",
    "1-1-2",
    "1-2-1",
    "1-2-2",
    "2-1-1",
    "2-1-2",
    "2-2-1",
    "2-2-2",
]

// Generate the slides and thumbnails dynamically
function generateSlides(taskName) {
    const slideshowContainer = $("#" + taskName + "-container");
    let slidesHTML = "";
    let thumbnailsHTML = "";
    let scenes = data[taskName]["data"];
    let use_20 = data[taskName]["use_20"];
    let use_two_sided = data[taskName]["use_two_sided"];

    if (use_20) {
        scenes.forEach((scene, sceneIndex) => {
            slidesHTML += `
                <div class="${taskName}-mySlides ${taskName}-scene-${sceneIndex} outer-container">
                    <div class="twenty-container">
            `;
            scene.images.forEach((image) => {
                slidesHTML += `
                    <img class="main-image" src="${image}" alt="${scene.title}">
                `;
            });
            slidesHTML += `
                    </div> 
                </div>
            `;

            thumbnailsHTML += `
                <div class="carousel-column">
                    <img class="${taskName}-demo cursor" src="${scene.images[0]}" data-scene-index="${sceneIndex}" alt="${scene.title}">
                </div>
            `;
        });

        slideshowContainer.html(`
            <div class="container" style="position: relative;">
                ${slidesHTML}

                <!-- Caption container -->
                <div class="caption-container">
                    <p id="${taskName}-caption">${scenes[currentSceneIndex[taskName]].title}</p>
                </div>

                <!-- Thumbnails -->
                // <div class="carousel-row">
                //     ${thumbnailsHTML}
                // </div>
            </div>
        `);
    }
    else if (use_two_sided) {
        let slidesHTML = '';
        let thumbnailsHTML = '';
        let currentScene = scenes[currentSceneIndex[taskName]];
        let first_caption = typeof currentScene.title === 'string' ? currentScene.title : currentScene.title[0];

        scenes.forEach((scene, sceneIndex) => {
            scene.images.forEach((image, index) => {
                slidesHTML += `
                    <div class="${taskName}-mySlides ${taskName}-scene-${sceneIndex} outer-container">
                        <img class="main-image" src="${image}" alt="${scene.title}">
                    </div>
                `;
            });

            if (taskName == 'pose') {
                thumbnailsHTML += `
                    <div class="carousel-column">
                        <img class="${taskName}-demo cursor" src="${scene.pose_image}" data-scene-index="${sceneIndex}" alt="${scene.title}">
                    </div>
                `;
            } else {
                thumbnailsHTML += `
                    <div class="carousel-column">
                        <img class="${taskName}-demo cursor" src="${scene.input_image}" data-scene-index="${sceneIndex}" alt="${scene.title}">
                    </div>
                `;
            }
        });

        if (taskName == 'tryon') {
            let garments = data[taskName]['garments'];
            
            thumbnailsHTML = `
                <div class="carousel-column">
                    <div class="carousel-row">
                        <img class="${taskName}-demo garmentA active cursor" src="${garments.upper[0]}" data-group="A" data-value="1">
                        <img class="${taskName}-demo garmentA cursor" src="${garments.upper[1]}" data-group="A" data-value="2">
                    </div>
                </div>
                <div class="carousel-column">
                    <div class="carousel-row">
                        <img class="${taskName}-demo garmentB active cursor" src="${garments.lower[0]}" data-group="B" data-value="1"">
                        <img class="${taskName}-demo garmentB cursor" src="${garments.lower[1]}" data-group="B" data-value="2">
                    </div>
                </div>
                <div class="carousel-column">
                    <div class="carousel-row">
                        <img class="${taskName}-demo garmentC active cursor" src="${garments.shoes[0]}" data-group="C" data-value="1">
                        <img class="${taskName}-demo garmentC cursor" src="${garments.shoes[1]}" data-group="C" data-value="2">
                    </div>
                </div>
                
            `;
        }

        const inputImageHTML = `
            <div class="col-sm-4" style="height: 100%; max-width: 50%">
                <img class="main-image" src="${currentScene.input_image}" alt="Input Image">
            </div>
        `;

        const carouselHTML = `
            <div class="col-sm-8 container">
                <div style="position: relative;">
                    ${slidesHTML}
                    ${scenes[0].images.length > 1 ? `
                        <a class="two-prev" data-task-name="${taskName}">❮</a>
                        <a class="two-next" data-task-name="${taskName}">❯</a>
                    ` : ''}
                </div>
            </div>
        `;

        slideshowContainer.html(`
            <div class="container">
                <div class="row" style="height: 500px; max-width: 100%; background: #f5f5f5; margin: 0; display: flex">
                    ${inputImageHTML}
                    ${carouselHTML}
                </div>
                <div class="caption-container">
                    <p id="${taskName}-caption">${first_caption}</p>
                </div>
                <div class="carousel-row">
                    ${thumbnailsHTML}
                </div>
            </div>
        `);
    }
    else {
        scenes.forEach((scene, sceneIndex) => {
            scene.images.forEach((image, index) => {
                slidesHTML += `
                    <div class="${taskName}-mySlides ${taskName}-scene-${sceneIndex} outer-container">
                        <div class="numbertext">${sceneIndex + 1}.${index + 1} / ${scene.images.length}</div>
                        <img class="main-image" src="${image}" alt="${scene.title}">
                    </div>
                `;
            });

            thumbnailsHTML += `
                <div class="carousel-column">
                    <img class="${taskName}-demo cursor" src="${scene.images[0]}" data-scene-index="${sceneIndex}" alt="${scene.title}">
                </div>
            `;
        });
        if (scenes[0].images.length > 1) { 
            slideshowContainer.html(`
                <div class="container" style="position: relative;">
                    ${slidesHTML}

                    <!-- Navigation arrows -->
                    <a class="prev" data-task-name="${taskName}">❮</a>
                    <a class="next" data-task-name="${taskName}">❯</a>

                    <!-- Caption container -->
                    <div class="caption-container">
                        <p id="${taskName}-caption">${scenes[currentSceneIndex[taskName]].title}</p>
                    </div>
                </div>
            `);
        }
        else{
            slideshowContainer.html(`
                <div class="container" style="position: relative;">
                    ${slidesHTML}

                    <!-- Caption container -->
                    <div class="caption-container">
                        <p id="${taskName}-caption">${scenes[currentSceneIndex[taskName]].title}</p>
                    </div>
                </div>
            `);
        }
    }

    // Initialize the first slide
    showSlides(slideIndex, taskName);
}

function plusSlides(n, taskName) {
    let scenes = data[taskName]["data"];
    if (n > 0) {
        console.log(currentSceneIndex[taskName])
        if (slideIndex === scenes[currentSceneIndex[taskName]].images.length) {
            slideIndex = 1;
        } else {
            slideIndex += n;
        }
    } else {
        if (slideIndex === 1) {
            slideIndex = scenes[currentSceneIndex[taskName]].images.length;
        } else {
            slideIndex += n;
        }
    }

    showSlides(slideIndex, taskName);
}

function showScene(sceneIndex, taskName) {
    currentSceneIndex[taskName] = sceneIndex;
    slideIndex = 1;
    showSlides(slideIndex, taskName);
}

function showSlides(n, taskName) {
    console.log("showSlides", n, taskName);
    let slides = $(`.${taskName}-mySlides`);
    let dots = $(`.${taskName}-demo`);
    let captionText = $(`#${taskName}-caption`);
    let scenes = data[taskName]["data"];
    let use_20 = data[taskName]["use_20"];
    let use_two_sided = data[taskName]["use_two_sided"];

    if (n > slides.length) { slideIndex = 1; }
    if (n < 1) { slideIndex = slides.length; }

    slides.hide();
    dots.removeClass("active");

    if (use_20) {
        let sceneSlides = $(`.${taskName}-scene-${currentSceneIndex[taskName]}`);
        sceneSlides.eq(slideIndex - 1).show();

        $(".twenty-container").twentytwenty();
        $(".twenty-container").css("height", "500px");
    } else if (use_two_sided) {
        // Update the input image on the left side
        let currentScene = scenes[currentSceneIndex[taskName]];
        $(`#${taskName}-container .col-sm-4 img`).attr('src', currentScene.input_image);
        
        // Show the current slide on the right side
        let sceneSlides = $(`.${taskName}-scene-${currentSceneIndex[taskName]}`);
        sceneSlides.eq(slideIndex - 1).show();
    } else {
        let sceneSlides = $(`.${taskName}-scene-${currentSceneIndex[taskName]}`);
        sceneSlides.eq(slideIndex - 1).show();
    }

    if (taskName == 'tryon') {
        let currentKey = sceneIndexToInput[Number(currentSceneIndex[taskName])];
        let a = currentKey.split("-")[0];
        let b = currentKey.split("-")[1];
        let c = currentKey.split("-")[2];

        $(`.tryon-demo[data-group="A"][data-value=${a}]`).addClass("active");
        $(`.tryon-demo[data-group="B"][data-value=${b}]`).addClass("active");
        $(`.tryon-demo[data-group="C"][data-value=${c}]`).addClass("active");
        
    } else {
        dots.eq(currentSceneIndex[taskName]).addClass("active");
    }
    
    if (typeof scenes[currentSceneIndex[taskName]].title === 'string') {
        captionText.html(scenes[currentSceneIndex[taskName]].title);
    } else {
        captionText.html(scenes[currentSceneIndex[taskName]].title[slideIndex - 1]);
    }
}

// Generate the slides on page load
$(document).ready(function() {
    tasks.forEach(task => {
        generateSlides(task);
    });

    // Tab Logic
    $('.nav-tabs a').on('click', function(e) {
        e.preventDefault();

        let targetId = $(this).attr('href');

        // Update active tab
        $('.nav-tabs li').removeClass('active');
        $(this).parent('li').addClass('active');

        // Update tab content
        $('.tab-pane').removeClass('in active').addClass('fade');
        $(targetId).addClass('in active').removeClass('fade');
        $('#teaser-container').addClass('in active');
        
        // split using "-" and get the first element
        let taskName = targetId.split("-")[0].replace("#", "");
        console.log("Tab clicked", taskName);

        $(`.${taskName}-mySlides`).each(function() {
            const $slide = $(this);
            const $container = $slide.find('.twenty-container');
            
            console.log($container)

            // Remove only the overlay within this slide
            $('.twentytwenty-overlay').remove();
            
            // Remove only the handle within this slide
            $('.twentytwenty-handle').remove();
            
            // bug still persists.
            // console.log($slide.find(`.${taskName}-wrapper`))
            
            // // Move the container outside the wrapper but keep it within image2pose-mySlides
            // $container.insertBefore($container.parent().parent());
            
            // Remove the empty wrapper within this slide
            // $slide.find('.twentytwenty-wrapper').remove();
            
            // // Clean up any remaining empty elements within this slide
            // $slide.find('.twentytwenty-wrapper').each(function() {
            //     if ($(this).children().length === 0) {
            //         $(this).remove();
            //     }
            // });
            
            // console.log($slide.find('.twenty-container'))
            // $slide.find('.twenty-container').twentytwenty();
            // $slide.find('.twenty-container').css("height", "500px");
        });
        
        $('.twenty-container').twentytwenty();
        $('.twenty-container').css("height", "500px");
        twenty_inited[targetId] = true;
    });

    // Thumbnail click event
    $(document).on('click', '.cursor', function() {
        let taskName = $(this).attr('class').split('-')[0];
        if (taskName == 'tryon') {
            let group = $(this).attr('data-group');
            let value = $(this).attr('data-value');
            garmentSelected[group] = value;
            let sceneIndex = tryonSceneIndex[`${garmentSelected.A}-${garmentSelected.B}-${garmentSelected.C}`]
            showScene(sceneIndex, taskName);
        } else {
            let sceneIndex = $(this).data('scene-index');
            showScene(sceneIndex, taskName);
        }
    });

    // Navigation arrows click event
    $(document).on('click', '.prev, .next, .two-prev, .two-next', function() {
        let taskName = $(this).data('task-name');
        let direction = ($(this).hasClass('prev') || $(this).hasClass('two-prev')) ? -1 : 1;
        plusSlides(direction, taskName);
    });
});
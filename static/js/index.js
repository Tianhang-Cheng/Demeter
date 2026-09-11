window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var PCA_BASE = "./static/interpolation";
var PCA_3D_BASE = "./static/interpolation_3d";
var PCA_3D_LEAF_BASE = "./static/interpolation_3d_leaf";
var PCA_3D_STEM_BASE = "./static/interpolation_3d_stem";
var currentSpecies = "Papaya"; // Default species
var currentSpecies3D = "Papaya"; // Default species for 3D
var currentSpecies3DLeaf = "Papaya"; // Default species for 3D leaf
var currentSpecies3DStem = "Papaya"; // Default species for 3D stem
// Map species to folder prefixes if needed in future; Papaya uses base paths
var speciesConfig = {
  Papaya: { base: "./static/interpolation/papaya" },
  Soybean: { base: "./static/interpolation/soybean" },
  Geranium: { base: "./static/interpolation/geranium" },
  Tobacco: { base: "./static/interpolation/tobacco" }
};

var speciesConfig3D = {
  Papaya: { base: "./static/interpolation_3d/papaya" },
  Soybean: { base: "./static/interpolation_3d/soybean" },
  Geranium: { base: "./static/interpolation_3d/geranium" },
  Tobacco: { base: "./static/interpolation_3d/tobacco" }
};

var speciesConfig3DLeaf = {
  Papaya: { base: "./static/interpolation_3d_leaf/papaya" },
  Soybean: { base: "./static/interpolation_3d_leaf/soybean" },
  Geranium: { base: "./static/interpolation_3d_leaf/geranium" },
  Tobacco: { base: "./static/interpolation_3d_leaf/tobacco" }
};

var speciesConfig3DStem = {
  Papaya: { base: "./static/interpolation_3d_stem/papaya" },
  Soybean: { base: "./static/interpolation_3d_stem/soybean" },
  Geranium: { base: "./static/interpolation_3d_stem/geranium" },
  Tobacco: { base: "./static/interpolation_3d_stem/tobacco" }
};

function updateSpeciesImage(species) {
  var imgPath = './static/images/' + species + '.jpg';
  var $img = $('#species-image');
  if ($img.length) {
    $img.attr('src', imgPath);
    $img.attr('alt', species);
  }
}
var NUM_INTERP_FRAMES = 50; // Change this to your desired max value

var interp_images = [];
var pca_images = {
  1: [],
  2: [],
  3: [],
  4: []
};

var pca_images_3d_leaf = {
  1: [],
  2: [],
  3: [],
  4: []
};

var pca_images_3d_stem = {
  1: [],
  2: [],
  3: [],
  4: []
};

function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function preloadPCAImagesForSpecies(species) {
  var base = speciesConfig[species] ? speciesConfig[species].base : PCA_BASE;
  for (var component = 1; component <= 4; component++) {
    pca_images[component] = [];
    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
      var path = base + '/pca' + component + '/' + String(i).padStart(6, '0') + '.jpg';
      pca_images[component][i] = new Image();
      pca_images[component][i].src = path;
    }
  }
}

function preloadPCAImagesForSpecies3DLeaf(species) {
  var base = speciesConfig3DLeaf[species] ? speciesConfig3DLeaf[species].base : PCA_3D_LEAF_BASE;
  for (var component = 1; component <= 4; component++) {
    pca_images_3d_leaf[component] = [];
    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
      var path = base + '/pca' + component + '/' + String(i).padStart(6, '0') + '.jpg';
      pca_images_3d_leaf[component][i] = new Image();
      pca_images_3d_leaf[component][i].src = path;
      // Handle error if image doesn't exist
      pca_images_3d_leaf[component][i].onerror = function() {
        // Image not found, will be handled in display function
      };
      // Handle successful load
      pca_images_3d_leaf[component][i].onload = function() {
        // Image loaded successfully
      };
    }
  }
}

function preloadPCAImagesForSpecies3DStem(species) {
  var base = speciesConfig3DStem[species] ? speciesConfig3DStem[species].base : PCA_3D_STEM_BASE;
  for (var component = 1; component <= 4; component++) {
    pca_images_3d_stem[component] = [];
    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
      var path = base + '/pca' + component + '/' + String(i).padStart(6, '0') + '.jpg';
      pca_images_3d_stem[component][i] = new Image();
      pca_images_3d_stem[component][i].src = path;
      // Handle error if image doesn't exist
      pca_images_3d_stem[component][i].onerror = function() {
        // Image not found, will be handled in display function
      };
      // Handle successful load
      pca_images_3d_stem[component][i].onload = function() {
        // Image loaded successfully
      };
    }
  }
}

function setInterpolationImage(i, wrapperId) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#' + wrapperId).empty().append(image);
}

function setPCAImage(i, componentNum) {
  var image = pca_images[componentNum][i];
  if (image) {
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#pca' + componentNum + '-image-wrapper').empty().append(image);
  } else {
    $('#pca' + componentNum + '-image-wrapper').html('<p>Image not found</p>');
  }
}

function setPCAImage3DLeaf(i, componentNum) {
  var image = pca_images_3d_leaf[componentNum][i];
  if (image && image.complete && image.naturalWidth !== 0) {
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#pca' + componentNum + '-image-wrapper-3d-leaf').empty().append(image);
  } else if (image && image.src) {
    // Image is loading, show loading state and set up onload handler
    $('#pca' + componentNum + '-image-wrapper-3d-leaf').html('<p>Loading...</p>');
    image.onload = function() {
      if (this.complete && this.naturalWidth !== 0) {
        this.ondragstart = function() { return false; };
        this.oncontextmenu = function() { return false; };
        $('#pca' + componentNum + '-image-wrapper-3d-leaf').empty().append(this);
      }
    };
  } else {
    $('#pca' + componentNum + '-image-wrapper-3d-leaf').html('<p>Image not available</p>');
  }
}

function setPCAImage3DStem(i, componentNum) {
  var image = pca_images_3d_stem[componentNum][i];
  if (image && image.complete && image.naturalWidth !== 0) {
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#pca' + componentNum + '-image-wrapper-3d-stem').empty().append(image);
  } else if (image && image.src) {
    // Image is loading, show loading state and set up onload handler
    $('#pca' + componentNum + '-image-wrapper-3d-stem').html('<p>Loading...</p>');
    image.onload = function() {
      if (this.complete && this.naturalWidth !== 0) {
        this.ondragstart = function() { return false; };
        this.oncontextmenu = function() { return false; };
        $('#pca' + componentNum + '-image-wrapper-3d-stem').empty().append(this);
      }
    };
  } else {
    $('#pca' + componentNum + '-image-wrapper-3d-stem').html('<p>Image not available</p>');
  }
}

// 3D Viewer functionality
var viewers3D = {};
var currentMeshes3D = {};

// Soybean 3D Viewer
var soybeanViewer = null;
var soybeanMesh = null;
var isMouseDown = false;
var mouseX = 0, mouseY = 0;
var rotationX = 0, rotationY = 0;

// Soybean Fitting Results viewers
var soyfitState = {
  currentId: '1',
  currentSpecies: 'soybean',
  pcd: null,
  mesh: null,
  viewers: {
    pcd: null,
    mesh: null
  },
  isDragging: { pcd: false, mesh: false },
  lastMouse: { x: 0, y: 0 },
  rotation: { pcdX: -1.5, pcdY: 0, meshX: -1.5, meshY: 0 }
};

function init3DViewer(componentNum) {
  var viewerId = 'pca' + componentNum + '-viewer-3d';
  var container = document.getElementById(viewerId);
  
  if (!container) return;
  
  // Clear any existing content
  container.innerHTML = '';
  
  // Create Three.js scene
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
  var renderer = new THREE.WebGLRenderer({ antialias: true });
  
  renderer.setSize(container.offsetWidth, container.offsetHeight);
  renderer.setClearColor(0xf9f9f9);
  container.appendChild(renderer.domElement);
  
  // Add lighting
  var ambientLight = new THREE.AmbientLight(0x404040, 0.6);
  scene.add(ambientLight);
  
  var directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(1, 1, 1);
  scene.add(directionalLight);
  
  // Position camera
  camera.position.z = 5;
  
  // Store viewer components
  viewers3D[componentNum] = {
    scene: scene,
    camera: camera,
    renderer: renderer,
    container: container
  };
  
  // Load initial mesh
  load3DMesh(componentNum, 0);
  
  // Start render loop
  animate3D(componentNum);
}

function load3DMesh(componentNum, frameIndex) {
  var viewer = viewers3D[componentNum];
  if (!viewer) return;
  
  // Remove existing mesh
  if (currentMeshes3D[componentNum]) {
    viewer.scene.remove(currentMeshes3D[componentNum]);
  }
  
  // For now, create a simple placeholder geometry
  // In a real implementation, you would load the actual PLY file
  var geometry = new THREE.BoxGeometry(1, 1, 1);
  var material = new THREE.MeshLambertMaterial({ 
    color: 0x00ff00,
    wireframe: false,
  });
  
  // Add some variation based on component and frame
  var scale = 0.5 + (frameIndex / 100) * 0.5;
  var rotation = (frameIndex / 100) * Math.PI * 2;
  
  geometry.scale(scale, scale, scale);
  
  var mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.y = rotation;
  
  viewer.scene.add(mesh);
  currentMeshes3D[componentNum] = mesh;
  
  // Add loading indicator
  var viewerId = 'pca' + componentNum + '-viewer-3d';
  var container = document.getElementById(viewerId);
  if (container.querySelector('.viewer-loading')) {
    container.querySelector('.viewer-loading').style.display = 'none';
  }
}

function animate3D(componentNum) {
  var viewer = viewers3D[componentNum];
  if (!viewer) return;
  
  requestAnimationFrame(function() { animate3D(componentNum); });
  
  // Rotate mesh slowly
  if (currentMeshes3D[componentNum]) {
    currentMeshes3D[componentNum].rotation.y += 0.01;
  }
  
  viewer.renderer.render(viewer.scene, viewer.camera);
}

function updateSpeciesImage3D(species) {
  var imgPath = './static/images/' + species + '.jpg';
  var $img = $('#species-image-3d');
  if ($img.length) {
    $img.attr('src', imgPath);
    $img.attr('alt', species);
  }
}

function updateSpeciesImage3DLeaf(species) {
  var imgPath = './static/images/' + species + '.jpg';
  var $img = $('#species-image-3d-leaf');
  if ($img.length) {
    $img.attr('src', imgPath);
    $img.attr('alt', species);
  }
}

function updateSpeciesImage3DStem(species) {
  var imgPath = './static/images/' + species + '.jpg';
  var $img = $('#species-image-3d-stem');
  if ($img.length) {
    $img.attr('src', imgPath);
    $img.attr('alt', species);
  }
}

function resize3DViewers() {
  for (var componentNum in viewers3D) {
    var viewer = viewers3D[componentNum];
    if (viewer && viewer.container) {
      var width = viewer.container.offsetWidth;
      var height = viewer.container.offsetHeight;
      viewer.camera.aspect = width / height;
      viewer.camera.updateProjectionMatrix();
      viewer.renderer.setSize(width, height);
    }
  }
}

// Soybean 3D Viewer Functions
function initSoybeanViewer() {
  var container = document.getElementById('soybean-3d-viewer');
  if (!container) return;
  
  // Clear any existing content
  container.innerHTML = '';
  
  // Create Three.js scene
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
  var renderer = new THREE.WebGLRenderer({ antialias: true });
  
  renderer.setSize(container.offsetWidth, container.offsetHeight);
  renderer.setClearColor(0xf9f9f9);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  
  // Add lighting
  var ambientLight = new THREE.AmbientLight(0x404040, 0.4);
  scene.add(ambientLight);
  
  var directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 5, 5);
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.width = 2048;
  directionalLight.shadow.mapSize.height = 2048;
  scene.add(directionalLight);
  
  // Add hemisphere light for softer lighting
  var hemisphereLight = new THREE.HemisphereLight(0x87CEEB, 0x98FB98, 0.3);
  scene.add(hemisphereLight);
  
  // Position camera
  camera.position.set(0, 0, 5);
  
  // Store viewer components
  soybeanViewer = {
    scene: scene,
    camera: camera,
    renderer: renderer,
    container: container
  };
  
  // Load soybean mesh
  loadSoybeanMesh();
  
  // Add mouse controls
  addSoybeanControls();
  
  // Start render loop
  animateSoybean();
}

function loadSoybeanMesh() {
  if (!soybeanViewer) return;
  
  // Remove existing mesh
  if (soybeanMesh) {
    soybeanViewer.scene.remove(soybeanMesh);
  }
  
  // Load PLY file
  var loader = new THREE.PLYLoader();
  loader.load('./static/interpolation_3d/soybean/pca1/test.ply', function(geometry) {
    // Create material
    var material = new THREE.MeshLambertMaterial({ 
      color: 0x228B22, // Forest green
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      vertexColors: true
    });
    
    // Create mesh
    soybeanMesh = new THREE.Mesh(geometry, material);
    soybeanMesh.castShadow = true;
    soybeanMesh.receiveShadow = true;
    
    // Center and scale the mesh
    geometry.computeBoundingBox();
    var center = geometry.boundingBox.getCenter(new THREE.Vector3());
    geometry.translate(-center.x, -center.y, -center.z);
    
    // Scale to fit in viewer
    var size = geometry.boundingBox.getSize(new THREE.Vector3());
    var maxDim = Math.max(size.x, size.y, size.z);
    var scale = 3 / maxDim;
    soybeanMesh.scale.setScalar(scale);
    
    soybeanViewer.scene.add(soybeanMesh);
    
    // Hide loading text
    var loadingText = soybeanViewer.container.querySelector('.viewer-loading');
    if (loadingText) {
      loadingText.style.display = 'none';
    }
  }, function(progress) {
    console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
  }, function(error) {
    console.error('Error loading PLY file:', error);
    // Show error message
    var loadingText = soybeanViewer.container.querySelector('.viewer-loading');
    if (loadingText) {
      loadingText.textContent = 'Error loading 3D model';
      loadingText.style.color = '#ff3860';
    }
  });
}

function addSoybeanControls() {
  var container = document.getElementById('soybean-3d-viewer');
  if (!container) return;
  
  // Mouse down
  container.addEventListener('mousedown', function(event) {
    isMouseDown = true;
    mouseX = event.clientX;
    mouseY = event.clientY;
    container.style.cursor = 'grabbing';
  });
  
  // Mouse up
  container.addEventListener('mouseup', function(event) {
    isMouseDown = false;
    container.style.cursor = 'grab';
  });
  
  // Mouse move
  container.addEventListener('mousemove', function(event) {
    if (!isMouseDown || !soybeanMesh) return;
    
    var deltaX = event.clientX - mouseX;
    var deltaY = event.clientY - mouseY;
    
    rotationY += deltaX * 0.01;
    rotationX += deltaY * 0.01;
    
    // Limit vertical rotation
    rotationX = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotationX));
    
    soybeanMesh.rotation.x = rotationX;
    soybeanMesh.rotation.y = rotationY;
    
    mouseX = event.clientX;
    mouseY = event.clientY;
  });
  
  // Mouse wheel for zoom
  container.addEventListener('wheel', function(event) {
    event.preventDefault();
    if (!soybeanViewer) return;
    
    var zoomSpeed = 0.1;
    var zoom = event.deltaY > 0 ? 1 + zoomSpeed : 1 - zoomSpeed;
    
    soybeanViewer.camera.position.multiplyScalar(zoom);
    
    // Limit zoom
    var distance = soybeanViewer.camera.position.length();
    if (distance < 1) {
      soybeanViewer.camera.position.normalize().multiplyScalar(1);
    } else if (distance > 20) {
      soybeanViewer.camera.position.normalize().multiplyScalar(20);
    }
  });
  
  // Mouse leave
  container.addEventListener('mouseleave', function(event) {
    isMouseDown = false;
    container.style.cursor = 'grab';
  });
}

function animateSoybean() {
  if (!soybeanViewer) return;
  
  requestAnimationFrame(animateSoybean);
  
  // Auto-rotate slowly when not interacting
  if (!isMouseDown && soybeanMesh) {
    soybeanMesh.rotation.y += 0.005;
  }
  
  soybeanViewer.renderer.render(soybeanViewer.scene, soybeanViewer.camera);
}

function resizeSoybeanViewer() {
  if (!soybeanViewer || !soybeanViewer.container) return;
  
  var width = soybeanViewer.container.offsetWidth;
  var height = soybeanViewer.container.offsetHeight;
  soybeanViewer.camera.aspect = width / height;
  soybeanViewer.camera.updateProjectionMatrix();
  soybeanViewer.renderer.setSize(width, height);
}

// ---------- Overview Image Viewer ----------
var overviewState = {
  currentIndex: 0,
  images: [],
  totalImages: 0,
  isLoaded: false,
  isPlaying: false,
  playInterval: null
};

function initOverviewViewer() {
  // Generate image list from 0.jpg to 59.jpg
  overviewState.images = [];
  for (var i = 0; i < 60; i++) {
    overviewState.images.push({
      src: './static/overview/' + i + '.jpg',
      index: i
    });
  }
  overviewState.totalImages = overviewState.images.length;
  overviewState.currentIndex = 0;
  overviewState.isLoaded = true;
  overviewState.isPlaying = false;
  overviewState.playInterval = null;
  
  // Load first image
  loadOverviewImage(0);
  
  // Setup navigation buttons
  setupOverviewNavigation();
  
  // Setup play/pause button
  setupOverviewPlayButton();
  
  // Add keyboard navigation
  setupOverviewKeyboardNavigation();
  
  // Start auto-play by default
  setTimeout(function() {
    console.log('Initial state before starting auto-play:', overviewState.isPlaying);
    startOverviewAutoPlay();
  }, 100);
}

function loadOverviewImage(index) {
  if (!overviewState.isLoaded || index < 0 || index >= overviewState.totalImages) {
    return;
  }
  
  var img = document.getElementById('overview-image');
  var counter = document.getElementById('overview-counter');
  var prevBtn = document.getElementById('overview-prev-btn');
  var nextBtn = document.getElementById('overview-next-btn');
  
  if (!img || !counter || !prevBtn || !nextBtn) {
    return;
  }
  
  // Update image source
  img.src = overviewState.images[index].src;
  img.alt = 'Overview Image ' + (index + 1);
  
  // Update counter
  counter.textContent = (index + 1) + ' / ' + overviewState.totalImages;
  
  // Update navigation buttons
  prevBtn.disabled = (index === 0);
  nextBtn.disabled = (index === overviewState.totalImages - 1);
  
  // Update current index
  overviewState.currentIndex = index;
}

function setupOverviewPlayButton() {
  var playBtn = document.getElementById('overview-play-btn');
  var playIcon = document.getElementById('overview-play-icon');
  
  if (!playBtn || !playIcon) {
    return;
  }
  
  playBtn.addEventListener('click', function() {
    console.log('Play button clicked, current state:', overviewState.isPlaying);
    if (overviewState.isPlaying) {
      // Pause auto-play
      stopOverviewAutoPlay();
    } else {
      // Start auto-play
      startOverviewAutoPlay();
    }
  });
}

function startOverviewAutoPlay() {
  if (overviewState.isPlaying) {
    return;
  }
  
  console.log('Starting auto-play, setting icon to pause');
  overviewState.isPlaying = true;
  var playIcon = document.getElementById('overview-play-icon');
  if (playIcon) {
    // Use setAttribute to force the class change
    playIcon.setAttribute('class', 'fas fa-pause');
    // Force a re-render by temporarily hiding and showing
    playIcon.style.display = 'none';
    playIcon.offsetHeight; // Trigger reflow
    playIcon.style.display = '';
    console.log('Icon set to pause:', playIcon.className);
    console.log('Icon element found:', playIcon);
  } else {
    console.log('Icon element NOT found!');
  }
  
  // Play 2 images per second (500ms interval)
  overviewState.playInterval = setInterval(function() {
    var nextIndex = (overviewState.currentIndex + 1) % overviewState.totalImages;
    loadOverviewImage(nextIndex);
  }, 500);
}

function stopOverviewAutoPlay() {
  if (!overviewState.isPlaying) {
    return;
  }
  
  console.log('Stopping auto-play, setting icon to play');
  overviewState.isPlaying = false;
  var playIcon = document.getElementById('overview-play-icon');
  if (playIcon) {
    // Use setAttribute to force the class change
    playIcon.setAttribute('class', 'fas fa-play');
    // Force a re-render by temporarily hiding and showing
    playIcon.style.display = 'none';
    playIcon.offsetHeight; // Trigger reflow
    playIcon.style.display = '';
    console.log('Icon set to play:', playIcon.className);
    console.log('Icon element found:', playIcon);
  } else {
    console.log('Icon element NOT found!');
  }
  
  if (overviewState.playInterval) {
    clearInterval(overviewState.playInterval);
    overviewState.playInterval = null;
  }
}

function setupOverviewNavigation() {
  var prevBtn = document.getElementById('overview-prev-btn');
  var nextBtn = document.getElementById('overview-next-btn');
  
  if (prevBtn) {
    prevBtn.addEventListener('click', function() {
      // Stop auto-play when manual navigation is used
      if (overviewState.isPlaying) {
        stopOverviewAutoPlay();
      }
      if (overviewState.currentIndex > 0) {
        loadOverviewImage(overviewState.currentIndex - 1);
      }
    });
  }
  
  if (nextBtn) {
    nextBtn.addEventListener('click', function() {
      // Stop auto-play when manual navigation is used
      if (overviewState.isPlaying) {
        stopOverviewAutoPlay();
      }
      if (overviewState.currentIndex < overviewState.totalImages - 1) {
        loadOverviewImage(overviewState.currentIndex + 1);
      }
    });
  }
}

function setupOverviewKeyboardNavigation() {
  document.addEventListener('keydown', function(event) {
    // Only handle keyboard navigation if the overview viewer is visible
    var overviewContainer = document.querySelector('.overview-viewer-container');
    if (!overviewContainer || !overviewState.isLoaded) {
      return;
    }
    
    // Check if the overview section is in viewport
    var rect = overviewContainer.getBoundingClientRect();
    var isVisible = rect.top < window.innerHeight && rect.bottom > 0;
    
    if (!isVisible) {
      return;
    }
    
    switch(event.key) {
      case 'ArrowLeft':
        event.preventDefault();
        // Stop auto-play when manual navigation is used
        if (overviewState.isPlaying) {
          stopOverviewAutoPlay();
        }
        if (overviewState.currentIndex > 0) {
          loadOverviewImage(overviewState.currentIndex - 1);
        }
        break;
      case 'ArrowRight':
        event.preventDefault();
        // Stop auto-play when manual navigation is used
        if (overviewState.isPlaying) {
          stopOverviewAutoPlay();
        }
        if (overviewState.currentIndex < overviewState.totalImages - 1) {
          loadOverviewImage(overviewState.currentIndex + 1);
        }
        break;
      case ' ':
        // Spacebar to toggle play/pause
        event.preventDefault();
        if (overviewState.isPlaying) {
          stopOverviewAutoPlay();
        } else {
          startOverviewAutoPlay();
        }
        break;
    }
  });
}

// ---------- 3D Reconstruction Viewer ----------
var reconstructionState = {
  currentFolder: '15_i',
  viewers: {
    gt: null,
    lsys: null,
    nksr: null,
    our: null
  },
  meshes: {
    gt: null,
    lsys: null,
    nksr: null,
    our: null
  },
  isDragging: {
    gt: false,
    lsys: false,
    nksr: false,
    our: false
  },
  lastMouse: { x: 0, y: 0 },
  rotation: {
    gtX: -1.5, gtY: 0,
    lsysX: -1.5, lsysY: 0,
    nksrX: -1.5, nksrY: 0,
    ourX: -1.5, ourY: 0
  }
};

function initReconstructionViewers() {
  // Create viewers for each method
  reconstructionState.viewers.gt = createReconstructionViewer('reconstruction-gt-viewer');
  reconstructionState.viewers.lsys = createReconstructionViewer('reconstruction-lsys-viewer');
  reconstructionState.viewers.nksr = createReconstructionViewer('reconstruction-nksr-viewer');
  reconstructionState.viewers.our = createReconstructionViewer('reconstruction-our-viewer');

  // Load initial meshes
  loadReconstructionMeshes();

  // Add controls
  addReconstructionControls('gt');
  addReconstructionControls('lsys');
  addReconstructionControls('nksr');
  addReconstructionControls('our');

  // Start animation loops
  animateReconstruction('gt');
  animateReconstruction('lsys');
  animateReconstruction('nksr');
  animateReconstruction('our');
}

function createReconstructionViewer(containerId) {
  var container = document.getElementById(containerId);
  if (!container) return null;
  
  container.innerHTML = '';
  
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(60, container.offsetWidth / container.offsetHeight, 0.01, 1000);
  var renderer = new THREE.WebGLRenderer({ antialias: true });
  
  renderer.setSize(container.offsetWidth, container.offsetHeight);
  renderer.setClearColor(0xf9f9f9);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // Add lighting - brighter setup
  var ambientLight = new THREE.AmbientLight(0x404040, 1.2);
  scene.add(ambientLight);
  
  var directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
  directionalLight.position.set(2, 4, 3);
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.width = 1024;
  directionalLight.shadow.mapSize.height = 1024;
  scene.add(directionalLight);
  
  // Add additional light sources for better illumination
  var directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.6);
  directionalLight2.position.set(-2, -4, -3);
  scene.add(directionalLight2);
  
  var hemisphereLight = new THREE.HemisphereLight(0x87CEEB, 0x98FB98, 0.4);
  scene.add(hemisphereLight);

  camera.position.set(0, 0, 3.5);
  
  return { scene: scene, camera: camera, renderer: renderer, container: container, object: null };
}

function reconstructionBasePath() {
  return './static/reconstruction_3d/soybean/' + reconstructionState.currentFolder + '/';
}

function loadReconstructionMeshes() {
  loadReconstructionMesh('gt', 'gt.ply');
  loadReconstructionMesh('lsys', 'Lsys.ply');
  loadReconstructionMesh('nksr', 'nksr.ply');
  loadReconstructionMesh('our', 'our.ply');
}

function loadReconstructionMesh(method, filename) {
  var viewer = reconstructionState.viewers[method];
  if (!viewer) return;
  
  // Remove existing mesh
  if (reconstructionState.meshes[method]) {
    viewer.scene.remove(reconstructionState.meshes[method]);
  }
  
  var loader = new THREE.PLYLoader();
  loader.load(reconstructionBasePath() + filename, function(geometry) {
    // Center and scale geometry
    geometry.computeBoundingBox();
    var center = geometry.boundingBox.getCenter(new THREE.Vector3());
    geometry.translate(-center.x, -center.y, -center.z);
    
    var size = geometry.boundingBox.getSize(new THREE.Vector3());
    var maxDim = Math.max(size.x, size.y, size.z);
    var scale = 2.0 / maxDim;
    geometry.scale(scale, scale, scale);
    
    var object;
    
    // Ground truth is a point cloud, others are meshes
    if (method === 'gt') {
      var material = new THREE.PointsMaterial({ 
        size: 0.01, 
        color: 0x888888,  // Gray color
        vertexColors: true 
      });
      object = new THREE.Points(geometry, material);
    } else {
      var material = new THREE.MeshStandardMaterial({ 
        color: 0x888888,  // Gray color for all methods
        metalness: 0.0,   // Reduced metalness for brighter appearance
        roughness: 0.3,   // Reduced roughness for more reflection
        side: THREE.DoubleSide,
        vertexColors: true
      });
      object = new THREE.Mesh(geometry, material);
      object.castShadow = true;
      object.receiveShadow = true;
    }
    
    // Apply initial rotation
    var rotXKey = method + 'X';
    var rotYKey = method + 'Y';
    object.rotation.x = reconstructionState.rotation[rotXKey];
    object.rotation.y = reconstructionState.rotation[rotYKey];
    
    reconstructionState.meshes[method] = object;
    viewer.object = object;
    viewer.scene.add(object);
    
    // Hide loading text
    var loading = viewer.container.querySelector('.viewer-loading');
    if (loading) loading.style.display = 'none';
    
  }, undefined, function(error) {
    console.error('Failed to load ' + filename, error);
    var loading = viewer.container.querySelector('.viewer-loading');
    if (loading) {
      loading.textContent = 'Failed to load ' + method;
      loading.style.color = '#ff3860';
    }
  });
}

function addReconstructionControls(method) {
  var viewer = reconstructionState.viewers[method];
  if (!viewer) return;
  
  var rotXKey = method + 'X';
  var rotYKey = method + 'Y';
  
  viewer.container.addEventListener('mousedown', function(e) {
    reconstructionState.isDragging[method] = true;
    reconstructionState.lastMouse.x = e.clientX;
    reconstructionState.lastMouse.y = e.clientY;
    viewer.container.style.cursor = 'grabbing';
  });
  
  viewer.container.addEventListener('mouseup', function() {
    reconstructionState.isDragging[method] = false;
    viewer.container.style.cursor = 'grab';
  });
  
  viewer.container.addEventListener('mouseleave', function() {
    reconstructionState.isDragging[method] = false;
    viewer.container.style.cursor = 'grab';
  });
  
  viewer.container.addEventListener('mousemove', function(e) {
    if (!reconstructionState.isDragging[method] || !viewer.object) return;
    
    var dx = e.clientX - reconstructionState.lastMouse.x;
    var dy = e.clientY - reconstructionState.lastMouse.y;
    
    reconstructionState.rotation[rotYKey] += dx * 0.01;
    reconstructionState.rotation[rotXKey] += dy * 0.01;
    reconstructionState.rotation[rotXKey] = Math.max(-Math.PI/2, Math.min(Math.PI/2, reconstructionState.rotation[rotXKey]));
    
    viewer.object.rotation.x = reconstructionState.rotation[rotXKey];
    viewer.object.rotation.y = reconstructionState.rotation[rotYKey];
    
    reconstructionState.lastMouse.x = e.clientX;
    reconstructionState.lastMouse.y = e.clientY;
  });
  
  viewer.container.addEventListener('wheel', function(e) {
    e.preventDefault();
    var zoom = (e.deltaY > 0) ? 1.1 : 0.9;
    viewer.camera.position.multiplyScalar(zoom);
    
    var distance = viewer.camera.position.length();
    if (distance < 1) viewer.camera.position.normalize().multiplyScalar(1);
    if (distance > 20) viewer.camera.position.normalize().multiplyScalar(20);
  }, { passive: false });
}

function animateReconstruction(method) {
  var viewer = reconstructionState.viewers[method];
  if (!viewer) return;
  
  function loop() {
    requestAnimationFrame(loop);
    
    // Auto-rotate when not dragging
    if (!reconstructionState.isDragging[method] && viewer.object) {
      viewer.object.rotation.z += 0.005;
    }
    
    viewer.renderer.render(viewer.scene, viewer.camera);
  }
  loop();
}

function resizeReconstructionViewers() {
  var methods = ['gt', 'lsys', 'nksr', 'our'];
  methods.forEach(function(method) {
    var viewer = reconstructionState.viewers[method];
    if (!viewer || !viewer.container) return;
    
    var width = viewer.container.offsetWidth;
    var height = viewer.container.offsetHeight;
    viewer.camera.aspect = width / height;
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.setSize(width, height);
  });
}

// ---------- Soybean Fitting Results ----------
function initSoyfitViewers() {
  // Set image src
  updateSoyfitImage();

  // Create viewers
  soyfitState.viewers.pcd = createBasicViewer('soyfit-pcd-viewer');
  soyfitState.viewers.mesh = createBasicViewer('soyfit-mesh-viewer');

  // Load initial contents
  loadSoyfitPCD();
  loadSoyfitMesh();

  // Controls
  addDragZoomControls('pcd');
  addDragZoomControls('mesh');

  // Animate
  animateSoyfit('pcd');
  animateSoyfit('mesh');
}

function soyfitBasePath() {
  return './static/fitting_3d/' + soyfitState.currentSpecies + '/' + soyfitState.currentId + '/';
}

function updateSoyfitImage() {
  var img = document.getElementById('soyfit-image');
  if (img) img.src = soyfitBasePath() + 'image.jpg';
}

function createBasicViewer(containerId) {
  var container = document.getElementById(containerId);
  if (!container) return null;
  container.innerHTML = '';
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(60, container.offsetWidth / container.offsetHeight, 0.01, 1000);
  var renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.offsetWidth, container.offsetHeight);
  renderer.setClearColor(0xf9f9f9);
  container.appendChild(renderer.domElement);

  var ambient = new THREE.AmbientLight(0x404040, 0.6);
  scene.add(ambient);
  var dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(2, 4, 3);
  scene.add(dir);

  camera.position.set(0, 0, 3.5);
  return { scene: scene, camera: camera, renderer: renderer, container: container, object: null };
}

function centerAndScaleGeometry(geometry, targetSize) {
  geometry.computeBoundingBox();
  var center = geometry.boundingBox.getCenter(new THREE.Vector3());
  geometry.translate(-center.x, -center.y, -center.z);
  var size = geometry.boundingBox.getSize(new THREE.Vector3());
  var maxDim = Math.max(size.x, size.y, size.z);
  var scale = (targetSize || 2.5) / maxDim;
  geometry.scale(scale, scale, scale);
}

function loadSoyfitPCD() {
  var v = soyfitState.viewers.pcd; if (!v) return;
  if (v.object) v.scene.remove(v.object);
  var loader = new THREE.PLYLoader();
  loader.load(soyfitBasePath() + 'gt_pcd.ply', function(geometry) {
    centerAndScaleGeometry(geometry, 2.0);
    var material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
    var points = new THREE.Points(geometry, material);
    // Apply initial rotation
    points.rotation.x = soyfitState.rotation.pcdX;
    points.rotation.y = soyfitState.rotation.pcdY;
    v.object = points;
    v.scene.add(points);
    var loading = v.container.querySelector('.viewer-loading');
    if (loading) loading.style.display = 'none';
  }, undefined, function(err) {
    console.error('Failed to load gt_pcd.ply', err);
    var loading = v.container.querySelector('.viewer-loading');
    if (loading) { loading.textContent = 'Failed to load point cloud'; loading.style.color = '#ff3860'; }
  });
}

function loadSoyfitMesh() {
  var v = soyfitState.viewers.mesh; if (!v) return;
  if (v.object) v.scene.remove(v.object);
  var loader = new THREE.PLYLoader();
  loader.load(soyfitBasePath() + 'mesh.ply', function(geometry) {
    centerAndScaleGeometry(geometry, 2.0);
    var material = new THREE.MeshStandardMaterial({ 
      vertexColors: true, 
      metalness: 0.05, 
      roughness: 0.9,
      side: THREE.DoubleSide,
    });
    var mesh = new THREE.Mesh(geometry, material);
    // Apply initial rotation
    mesh.rotation.x = soyfitState.rotation.meshX;
    mesh.rotation.y = soyfitState.rotation.meshY;
    v.object = mesh;
    v.scene.add(mesh);
    var loading = v.container.querySelector('.viewer-loading');
    if (loading) loading.style.display = 'none';
  }, undefined, function(err) {
    console.error('Failed to load mesh.ply', err);
    var loading = v.container.querySelector('.viewer-loading');
    if (loading) { loading.textContent = 'Failed to load mesh'; loading.style.color = '#ff3860'; }
  });
}

function addDragZoomControls(kind) {
  var v = soyfitState.viewers[kind]; if (!v) return;
  var dragKey = kind === 'pcd' ? 'pcd' : 'mesh';
  var rotXKey = kind === 'pcd' ? 'pcdX' : 'meshX';
  var rotYKey = kind === 'pcd' ? 'pcdY' : 'meshY';

  v.container.addEventListener('mousedown', function(e){
    soyfitState.isDragging[dragKey] = true; soyfitState.lastMouse.x = e.clientX; soyfitState.lastMouse.y = e.clientY; v.container.style.cursor = 'grabbing';
  });
  v.container.addEventListener('mouseup', function(){ soyfitState.isDragging[dragKey] = false; v.container.style.cursor = 'grab'; });
  v.container.addEventListener('mouseleave', function(){ soyfitState.isDragging[dragKey] = false; v.container.style.cursor = 'grab'; });
  v.container.addEventListener('mousemove', function(e){
    if (!soyfitState.isDragging[dragKey] || !v.object) return;
    var dx = e.clientX - soyfitState.lastMouse.x; var dy = e.clientY - soyfitState.lastMouse.y;
    soyfitState.rotation[rotYKey] += dx * 0.01; soyfitState.rotation[rotXKey] += dy * 0.01;
    soyfitState.rotation[rotXKey] = Math.max(-Math.PI/2, Math.min(Math.PI/2, soyfitState.rotation[rotXKey]));
    v.object.rotation.x = soyfitState.rotation[rotXKey]; v.object.rotation.y = soyfitState.rotation[rotYKey];
    soyfitState.lastMouse.x = e.clientX; soyfitState.lastMouse.y = e.clientY;
  });
  v.container.addEventListener('wheel', function(e){
    e.preventDefault();
    var s = (e.deltaY > 0) ? 1.1 : 0.9; v.camera.position.multiplyScalar(s);
    var d = v.camera.position.length();
    if (d < 1) v.camera.position.normalize().multiplyScalar(1);
    if (d > 20) v.camera.position.normalize().multiplyScalar(20);
  }, { passive: false });
}

function animateSoyfit(kind) {
  var v = soyfitState.viewers[kind]; if (!v) return;
  function loop(){
    requestAnimationFrame(loop);
    if (!soyfitState.isDragging[kind] && v.object) { v.object.rotation.z += 0.005; }
    v.renderer.render(v.scene, v.camera);
  }
  loop();
}

function resizeSoyfitViewers() {
  ['pcd','mesh'].forEach(function(kind){
    var v = soyfitState.viewers[kind]; if (!v || !v.container) return;
    var w = v.container.offsetWidth; var h = v.container.offsetHeight;
    v.camera.aspect = w / h; v.camera.updateProjectionMatrix();
    v.renderer.setSize(w, h);
  });
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();
    preloadPCAImagesForSpecies(currentSpecies);
    preloadPCAImagesForSpecies3DLeaf(currentSpecies3DLeaf);
    preloadPCAImagesForSpecies3DStem(currentSpecies3DStem);
    updateSpeciesImage(currentSpecies);

    // Initialize PCA component sliders
    for (var i = 1; i <= 4; i++) {
      var sliderId = '#pca' + i + '-slider';
      $(sliderId).on('input', function(event) {
        var componentNum = this.id.replace('pca', '').replace('-slider', '');
        setPCAImage(this.value, componentNum);
      });
      setPCAImage(0, i);
      $(sliderId).prop('max', NUM_INTERP_FRAMES - 1);
    }

    // Initialize 3D PCA component viewers
    updateSpeciesImage3D(currentSpecies3D);
    for (var i = 1; i <= 4; i++) {
      // Initialize 3D viewer
      init3DViewer(i);
      
      // Set up slider
      var sliderId = '#pca' + i + '-slider-3d';
      $(sliderId).on('input', function(event) {
        var componentNum = this.id.replace('pca', '').replace('-slider-3d', '');
        load3DMesh(componentNum, this.value);
      });
      $(sliderId).prop('max', NUM_INTERP_FRAMES - 1);
    }

    // Initialize 3D Leaf PCA component sliders
    updateSpeciesImage3DLeaf(currentSpecies3DLeaf);
    for (var i = 1; i <= 4; i++) {
      var sliderId = '#pca' + i + '-slider-3d-leaf';
      $(sliderId).on('input', function(event) {
        var componentNum = this.id.replace('pca', '').replace('-slider-3d-leaf', '');
        setPCAImage3DLeaf(this.value, componentNum);
      });
      $(sliderId).prop('max', NUM_INTERP_FRAMES - 1);
      setPCAImage3DLeaf(0, i);
    }

    // Initialize 3D Stem PCA component sliders
    updateSpeciesImage3DStem(currentSpecies3DStem);
    for (var i = 1; i <= 4; i++) {
      var sliderId = '#pca' + i + '-slider-3d-stem';
      $(sliderId).on('input', function(event) {
        var componentNum = this.id.replace('pca', '').replace('-slider-3d-stem', '');
        setPCAImage3DStem(this.value, componentNum);
      });
      $(sliderId).prop('max', NUM_INTERP_FRAMES - 1);
      setPCAImage3DStem(0, i);
    }

    bulmaSlider.attach();

    // Species selector handlers
    $('.species-box').on('click', function() {
      var selected = $(this).data('species');
      if (selected === currentSpecies) return;
      currentSpecies = selected;

      // Highlight active
      $('.species-box').removeClass('is-active');
      $(this).addClass('is-active');

      // Update label
      $('#species-label').text(currentSpecies);
      updateSpeciesImage(currentSpecies);

      // Reload images for selected species and reset sliders to 0
      preloadPCAImagesForSpecies(currentSpecies);
      for (var c = 1; c <= 4; c++) {
        var sId = '#pca' + c + '-slider';
        $(sId).val(0);
        setPCAImage(0, c);
      }
    });

    // 3D Species selector handlers
    $('.species-box-3d').on('click', function() {
      var selected = $(this).data('species');
      if (selected === currentSpecies3D) return;
      currentSpecies3D = selected;

      // Highlight active
      $('.species-box-3d').removeClass('is-active');
      $(this).addClass('is-active');

      // Update label
      $('#species-label-3d').text(currentSpecies3D);
      updateSpeciesImage3D(currentSpecies3D);

      // Reset 3D sliders to 0 and reload meshes
      for (var c = 1; c <= 4; c++) {
        var sId = '#pca' + c + '-slider-3d';
        $(sId).val(0);
        load3DMesh(c, 0);
      }
    });

    // 3D Leaf Species selector handlers
    $('.species-box-3d-leaf').on('click', function() {
      var selected = $(this).data('species');
      if (selected === currentSpecies3DLeaf) return;
      currentSpecies3DLeaf = selected;

      // Highlight active
      $('.species-box-3d-leaf').removeClass('is-active');
      $(this).addClass('is-active');

      // Update label
      $('#species-label-3d-leaf').text(currentSpecies3DLeaf);
      updateSpeciesImage3DLeaf(currentSpecies3DLeaf);

      // Reload images for selected species and reset sliders to 0
      preloadPCAImagesForSpecies3DLeaf(currentSpecies3DLeaf);
      for (var c = 1; c <= 4; c++) {
        var sId = '#pca' + c + '-slider-3d-leaf';
        $(sId).val(0);
        setPCAImage3DLeaf(0, c);
      }
    });

    // 3D Stem Species selector handlers
    $('.species-box-3d-stem').on('click', function() {
      var selected = $(this).data('species');
      if (selected === currentSpecies3DStem) return;
      currentSpecies3DStem = selected;

      // Highlight active
      $('.species-box-3d-stem').removeClass('is-active');
      $(this).addClass('is-active');

      // Update label
      $('#species-label-3d-stem').text(currentSpecies3DStem);
      updateSpeciesImage3DStem(currentSpecies3DStem);

      // Reload images for selected species and reset sliders to 0
      preloadPCAImagesForSpecies3DStem(currentSpecies3DStem);
      for (var c = 1; c <= 4; c++) {
        var sId = '#pca' + c + '-slider-3d-stem';
        $(sId).val(0);
        setPCAImage3DStem(0, c);
      }
    });

    // Handle window resize for 3D viewers
    $(window).on('resize', function() {
      resize3DViewers();
      resizeSoybeanViewer();
      resizeSoyfitViewers();
      resizeReconstructionViewers();
    });

    // Initialize soybean 3D viewer
    initSoybeanViewer();

    // Initialize Soybean Fitting Results
    initSoyfitViewers();

    // Wire soybean fit buttons
    $('.soybean-fit-btn').on('click', function(){
      var id = $(this).data('fit-id').toString();
      if (id === soyfitState.currentId && soyfitState.currentSpecies === 'soybean') return;
      $('.soybean-fit-btn').removeClass('is-active');
      $('.ribes-fit-btn').removeClass('is-active');
      $('.maize-fit-btn').removeClass('is-active');
      $('.tobacco-fit-btn').removeClass('is-active');
      $(this).addClass('is-active');
      soyfitState.currentId = id;
      soyfitState.currentSpecies = 'soybean';
      updateSoyfitImage();
      // Reset rotations
      soyfitState.rotation = { pcdX: -1.5, pcdY: 0, meshX: -1.5, meshY: 0 };
      loadSoyfitPCD();
      loadSoyfitMesh();
    });

    // Wire ribes fit buttons
    $('.ribes-fit-btn').on('click', function(){
      var id = $(this).data('fit-id').toString();
      if (id === soyfitState.currentId && soyfitState.currentSpecies === 'ribes') return;
      $('.soybean-fit-btn').removeClass('is-active');
      $('.ribes-fit-btn').removeClass('is-active');
      $('.maize-fit-btn').removeClass('is-active');
      $('.tobacco-fit-btn').removeClass('is-active');
      $(this).addClass('is-active');
      soyfitState.currentId = id;
      soyfitState.currentSpecies = 'ribes';
      updateSoyfitImage();
      // Reset rotations
      soyfitState.rotation = { pcdX: -1.5, pcdY: 0, meshX: -1.5, meshY: 0 };
      loadSoyfitPCD();
      loadSoyfitMesh();
    });

    // Wire maize fit buttons
    $('.maize-fit-btn').on('click', function(){
      var id = $(this).data('fit-id').toString();
      if (id === soyfitState.currentId && soyfitState.currentSpecies === 'maize') return;
      $('.soybean-fit-btn').removeClass('is-active');
      $('.ribes-fit-btn').removeClass('is-active');
      $('.maize-fit-btn').removeClass('is-active');
      $('.tobacco-fit-btn').removeClass('is-active');
      $(this).addClass('is-active');
      soyfitState.currentId = id;
      soyfitState.currentSpecies = 'maize';
      updateSoyfitImage();
      // Reset rotations
      soyfitState.rotation = { pcdX: -1.5, pcdY: 0, meshX: -1.5, meshY: 0 };
      loadSoyfitPCD();
      loadSoyfitMesh();
    });

    // Wire tobacco fit buttons
    $('.tobacco-fit-btn').on('click', function(){
      var id = $(this).data('fit-id').toString();
      if (id === soyfitState.currentId && soyfitState.currentSpecies === 'tobacco') return;
      $('.soybean-fit-btn').removeClass('is-active');
      $('.ribes-fit-btn').removeClass('is-active');
      $('.maize-fit-btn').removeClass('is-active');
      $('.tobacco-fit-btn').removeClass('is-active');
      $(this).addClass('is-active');
      soyfitState.currentId = id;
      soyfitState.currentSpecies = 'tobacco';
      updateSoyfitImage();
      // Reset rotations
      soyfitState.rotation = { pcdX: -1.5, pcdY: 0, meshX: -1.5, meshY: 0 };
      loadSoyfitPCD();
      loadSoyfitMesh();
    });

    // Initialize Overview Image Viewer
    initOverviewViewer();

    // Initialize 3D Reconstruction Viewers
    initReconstructionViewers();

    // Wire reconstruction folder buttons
    $('.reconstruction-folder-btn').on('click', function(){
      var folder = $(this).data('folder');
      if (folder === reconstructionState.currentFolder) return;
      
      $('.reconstruction-folder-btn').removeClass('is-active');
      $(this).addClass('is-active');
      reconstructionState.currentFolder = folder;
      
      // Reset rotations
      reconstructionState.rotation = {
        gtX: -1.5, gtY: 0,
        lsysX: -1.5, lsysY: 0,
        nksrX: -1.5, nksrY: 0,
        ourX: -1.5, ourY: 0
      };
      
      // Reload all meshes
      loadReconstructionMeshes();
    });

})


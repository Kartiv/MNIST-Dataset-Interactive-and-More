class jn{

    static random(a,b){
        return a + Math.random()*(b-a);
    }
    
    static randint(a,b){
        return a+Math.floor(Math.random()*(b-a));
    }
    
    static randomize(arr){
        let newarr = [];
        let m = arr.length;
        for(let i=0; i<m; i++){
            let s = jn.randint(0,arr.length);
            newarr.push(arr.splice(s,1)[0]);
        }
        return newarr;
    }

    static dot(lst1, lst2){
        let s=0;
        for(let i=0; i<lst1.length; i++){
            s+=lst1[i] * lst2[i]
        }
        return s
    }

    static operator(A, b){ //matrix A times vector b
        let C = [];
        for(let r=0; r<A.length; r++){
            C.push(jn.dot(A[r], b));
        }
        return C
    }

    static areEqual(a,b){
        if(a.length!=b.length){
            return false;
        }
        for(let i=0; i<a.length; i++){
            if(a[i]!=b[i]){
                return false;
            }
        }
        return true;
    }

    static randArr(n){
        let lst = [];
        for(let i=0; i<n; i++){
            lst.push(jn.random(-1, 1));
        }
        return lst;
    }

    static roundArr(arr){
        let newArr = [];
        for(let i=0; i<arr.length; i++){
            newArr.push(Math.round(arr[i]));
        }
        return newArr;
    }

    static sigmoid(z){
        //return 1/(1+Math.exp(-z));
        return Math.atan(z)/Math.PI+1/2;
    }

    static lstSigmoid(lst){
        var nlist = [];
        for(let i=0; i<lst.length; i++){
            nlist.push(jn.sigmoid(lst[i]));
        }
        return nlist;
    }

    //POLYGONS

    static generateConvex(N, bound){
        let X = [];
        let Y = [];
        for(let i=0; i<N; i++){
            X[i] = jn.randint(0,bound);
            Y[i] = jn.randint(0,bound);
        }
        X.sort((a,b)=>{
            return a-b;
        })
        Y.sort((a,b)=>{
            return a-b;
        })
        let xmin = X[0];
        let xmax = X[X.length-1];
        let ymin = Y[0];
        let ymax = Y[Y.length-1];
        let xGroups = [[xmin],[xmin]];
        let yGroups = [[ymin], [ymin]];
        for(let i=1; i<N-1; i++){
            let s1 = jn.randint(0,2);
            let s2 = jn.randint(0,2);
            if(s1){
                xGroups[0].push(X[i])
            }
            else{
                xGroups[1].push(X[i]);
            }
            if(s2){
                yGroups[0].push(Y[i])
            }
            else{
                yGroups[1].push(Y[i]);
            }
        }
        xGroups[0].push(xmax);
        xGroups[1].push(xmax);
        yGroups[0].push(ymax);
        yGroups[1].push(ymax);

        let xVec = [];
        let yVec = [];
        for(let i=0; i<xGroups[0].length-1; i++){
            xVec.push(xGroups[0][i+1]-xGroups[0][i]);
        }
        for(let i=0; i<xGroups[1].length-1; i++){
            xVec.push(xGroups[1][i]-xGroups[1][i+1]);
        }
        for(let i=0; i<yGroups[0].length-1; i++){
            yVec.push(yGroups[0][i+1]-yGroups[0][i]);
        }
        for(let i=0; i<yGroups[1].length-1; i++){
            yVec.push(yGroups[1][i]-yGroups[1][i+1]);
        }

        yVec = jn.randomize(yVec);
        
        let Vectors = [];
        for(let i=0; i<xVec.length; i++){
            Vectors.push(new vec2d(xVec[i], yVec[i]));
        }
        
        Vectors.sort((a,b)=>{
            let anga = Math.atan2(a.x1, a.x0);
            let angb = Math.atan2(b.x1, b.x0);
            if(anga<0){
                anga+=2*Math.PI;
            }
            if(angb<0){
                angb+=2*Math.PI;
            }
            return anga-angb;
        })

        let verts = [Vectors[0].add(new vec2d(xmax,ymax))];
        for(let i=1; i<Vectors.length; i++){
            verts.push(verts[i-1].add(Vectors[i]));
        }

        return new polygon(verts);
    }

    static createRect(x,y,width,height){
        return new polygon([new vec2d(x-width/2, y-height/2), new vec2d(x-width/2, y+height/2), new vec2d(x+width/2, y-height/2),
            new vec2d(x+width/2, y+height/2)]);
    }

    static SAT(poly1, poly2){
        for(let i=0; i<poly1.vertices.length; i++){
            let axis = poly1.edge(i).normal();
            let p1 = poly1.project(axis);
            let p2 = poly2.project(axis);
            if(p1[0]>p2[1] || p2[0]>p1[1]){
                return false;
            }
        }
        for(let i=0; i<poly2.vertices.length; i++){
            let axis = poly2.edge(i).normal();
            let p1 = poly1.project(axis);
            let p2 = poly2.project(axis);
            if(p1[0]>p2[1] || p2[0]>p1[1]){
                return false;
            }
        }
        return true;
    }
}

class Neural_Network{

    constructor(system, data, target, niter = 1000, alpha = 1, thetas = null){
        this.system = system;
        this.data = data;
        this.target = target;
        this.niter = niter;
        this.alpha = alpha;
        this.thetas = thetas;
    }

    generateThetas(){
        if(!thetas){
            var thetas = [];
            for(let layer=1; layer<this.system.length; layer++){
                thetas.push([]);
                for(let neuron=0; neuron<this.system[layer]; neuron++){
                    thetas[thetas.length-1].push(jn.randArr(this.system[layer-1]+1));
                }
            }
            this.thetas = thetas;
        }
    }

    forwardProp(x0){
        let t = [];
        for(let i=0; i<x0.length; i++){
            t.push(x0[i]);
        }
        t.splice(0,0,1);
        var values = [t];
        for(let layer=1; layer<this.system.length; layer++){
            let temp = jn.lstSigmoid(jn.operator(this.thetas[layer-1], values[layer-1]));
            temp.splice(0,0,1);
            values.push(temp);
        }
        return values
    }

    backProp(){
        for(let point=0; point<this.data.length; point++){ //for each data point
            let values = this.forwardProp(this.data[point]); //calculate all of the values
            let deltas = [[]]; //first (last) layer of deltas
            for(let i=1; i<values[values.length-1].length; i++){ //ignore bias which is the first element
                deltas[0].push(-2*(this.target[point][i-1]-values[values.length-1][i])*values[values.length-1][i]*(1-values[values.length-1][i])) //definition of deltas
                for(let j=0; j<values[values.length-2].length; j++){
                    this.thetas[this.thetas.length-1][i-1][j]-=this.alpha * deltas[0][deltas[0].length-1]*values[values.length-2][j] //gradient descent
                }
            }

            for(let layer=values.length-2; layer>0; layer--){ //dont include first layer because we already calculated it, dont include last because its the input layer
                deltas.push([]);
                for(let i=1; i<values[layer].length; i++){ //ignore bias, doesnt matter for backwards propagation
                    let s = 0; //calulate dot product by hand because numpy can go suck a fucking lollipop. this is the dot product/sum in definition for deltas
                    for(let k=1; k<values[layer+1].length; k++){ //once again, fuck the bias
                        s+=deltas[deltas.length-2][k-1] * this.thetas[layer][k-1][i] //deltas doesnt include a bias (starts from index 1), so we k-1 instead of k. additionaly, thetas array is made so thetas[layer]
                    } //are the thetas for the next layer, none go to bias, 2d arr which is practically a single dim array so we take 0 coord, ith neuron is what we are looking for
                    deltas[deltas.length-1].push(s*values[layer][i]*(1-values[layer][i])); 
                    for(let j=0; j<values[layer-1].length; j++){//adjust all of the thetas
                        this.thetas[layer-1][i-1][j] -= this.alpha * deltas[deltas.length-1][deltas[deltas.length-1].length-1]*values[layer-1][j]; //read
                    }
                }
            }
        }
    }

    solve(){
        this.generateThetas();
        for(let run=0; run<this.niter; run++){
            this.backProp();
        }        
    }
    

    evaluate(){
        let count = 0;
        for(let i=0; i<this.data.length; i++){
            if (jn.areEqual(jn.roundArr(this.predict(this.data[i])), this.target[i])){
                count++;
            }
        }
        return {"count": count, "successRate": count/this.data.length}
    }

    predict(x0){
        let temp = this.forwardProp(x0)
        temp = temp[temp.length-1];
        temp.splice(0,1);
        return temp;
    }
}

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d');
canvas.width = 28*3;
canvas.height = 28*3;

var inputArray;

var lastX;
var lastY;
var mouseX;
var mouseY;
var mouseX;
var mouseY;
var isDrawing = false;

var boundingBox;

var interval;

var evalFlag = false;

const gray = 0.2;

// var data = [[0,0], [1,0], [0,1], [1,1]];
// var target = [[1], [0], [0], [1]];
// var network = new Neural_Network([2, 3, 1], data, target);
// network.solve();
// console.log(network.evaluate())

const predictDiv = document.getElementById('prediction');

function setEventListeners(){
    canvas.addEventListener("mousedown", mouseDownListener);
    canvas.addEventListener("mouseup", mouseUpListener);
    canvas.addEventListener("mousemove", cmouseMoveListener);
    document.addEventListener("mousemove", mouseMoveListener);
}

function clearEventListeners(){
    canvas.removeEventListener("mousedown", mouseDownListener);
    canvas.removeEventListener("mouseup", mouseUpListener);
    canvas.removeEventListener("mousemove", cmouseMoveListener);
    document.removeEventListener("mousemove", mouseMoveListener);
}

function cmouseMoveListener(event){
    lastX = mouseX;
    lastY = mouseY;
    mouseX = event.offsetX;
    mouseY = event.offsetY;
}

function mouseMoveListener(event){
    if(event.offsetX>canvas.width || event.offsetY > canvas.height){
        isDrawing = false;
    }
}

function mouseDownListener(event){
    isDrawing = true;
    boundingBox = [mouseX, mouseX, mouseY, mouseY];
}

function mouseUpListener(event){
    isDrawing = false;
    
}

function resetListener(event){
    if(event.key==' '){
        ctx.clearRect(0,0,canvas.width, canvas.height);
        start();
    }
}

function start(){
    clearEventListeners();
    setEventListeners();
    inputArray = [];
    for(let i=0; i<28*2; i++){
        inputArray.push([]);
        for(let j=0; j<28*2; j++){
            inputArray[i].push(0);
        }
    }
    interval = setInterval(()=>{
        let inp = network.predict(convert(inputArray));
        predictDiv.innerText = (argmax(inp).toString()) + " | Certainty: " + inp[argmax(inp)].toFixed(4).toString();
        draw(28*2);
    }, 1)
}

function convert(input){
    lowRes = []
    for(let i=0; i<input.length-1; i+=2){
        for(let j=0; j<input.length-1; j+=2){
            lowRes.push((input[i][j] + input[i][j+1] + input[i+1][j]+input[i+1][j+1])/4);
        }
    }
    return lowRes;
}

function argmax(arr){
    let index = 0;
    let max = -10;
    for(let i=0; i<arr.length; i++){
        if(arr[i]>max){
            index = i;
            max = arr[i]
        }
    }
    return index;
}

function draw(c){
    if(isDrawing){

        if(mouseX<boundingBox[0]){
            boundingBox[0] = mouseX;
        }
        else if(mouseX>boundingBox[1]){
            boundingBox[1] = mouseX;
        }
        if(mouseY<boundingBox[2]){
            boundingBox[2] = mouseY;
        }
        else if(mouseY>boundingBox[3]){
            boundingBox[3] = mouseY;
        }

        for(let t=0; t<1; t+=0.05){
            let tx = lastX + t*(mouseX - lastX);
            let ty = lastY + t*(mouseY - lastY);

            let x = tx - tx%(canvas.width/c);
            let y = ty - ty%(canvas.height/c);

            let iy = x/(canvas.width/c);
            let ix = y/(canvas.height/c);

            if(inputArray[ix][iy]+2*gray >=1){
                inputArray[ix+1][iy] = 1;
            }
            else{
                inputArray[ix+1][iy] += 2*gray;
            }

            for(let i=1; i<3; i++){
                if(inputArray[ix+i][iy]+gray >=1){
                    inputArray[ix+i][iy] =1;
                }
                else{
                    inputArray[ix+i][iy] += gray;
                }
    
                if(inputArray[ix-i][iy]+gray >=1){
                    inputArray[ix-i][iy] =1;
                }
                else{
                    inputArray[ix-i][iy] += gray;
                }
    
                if(inputArray[ix][iy+i]+gray >=1){
                    inputArray[ix][iy+i] =1;
                }
                else{
                    inputArray[ix][iy+i] += gray;
                }
    
                if(inputArray[ix][iy-i]+gray >=1){
                    inputArray[ix][iy-i] =1;
                }
                else{
                    inputArray[ix][iy-i] += gray;
                }

                ctx.fillStyle = 'hsl(0,0%,0)';
                ctx.fillRect(x,y,canvas.width/c, canvas.width/c);
                ctx.fill();
    
                ctx.fillStyle = 'hsl(0,0%,' + ((1-inputArray[ix+i][iy])*100) + '%)';
                ctx.fillRect(x+i*canvas.width/c,y,canvas.width/c, canvas.width/c);
                ctx.fill();
    
                ctx.fillStyle = 'hsl(0,0%,' + ((1-inputArray[ix-i][iy])*100) + '%)';
                ctx.fillRect(x-i*canvas.width/c,y,canvas.width/c, canvas.width/c);
                ctx.fill();
    
                ctx.fillStyle = 'hsl(0,0%,' + ((1-inputArray[ix][iy+i])*100) + '%)';
                ctx.fillRect(x,y+i*canvas.height/c,canvas.width/c, canvas.width/c);
                ctx.fill();
    
                ctx.fillStyle = 'hsl(0,0%,' + ((1-inputArray[ix][iy-i])*100) + '%)';
                ctx.fillRect(x,y-i*canvas.height/c,canvas.width/c, canvas.width/c);
                ctx.fill();
            }
        }
    }
    // for(let i=0; i<inputArray.length; i++){
    //     for(let j=0; j<inputArray.length; j++){
    //         ctx.fillStyle = 'hsl(0,0%,' + ((1-inputArray[i][j])*100) + '%)';
    //         ctx.fillRect(i*canvas.width/28, j*canvas.height/28, canvas.width/28, canvas.height/28);
    //         ctx.fill();
    //     }
    // }
}

const Thetas = JSON.parse(localStorage.getItem('thetas'));

var network = new Neural_Network([784, 100, 10], [], [], 0, 0, Thetas)

document.addEventListener('keypress', resetListener);

start();



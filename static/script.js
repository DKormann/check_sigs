

let table = document.querySelector('table');

function sigmoid(x){
  return 1/(1+Math.exp(-x));
}

for (let index = 0; index < 20; index++) {
    let row = table.insertRow();
    sample_n = Math.floor(Math.random() * 600)
    for (let k = 0; k < 2; k++) {
      let img = document.createElement('img');
      img.src = `image/${k?'r':'f'}/${sample_n}`;
      let cell = row.insertCell();
      cell.appendChild(img);
    }
    let celllabel = row.insertCell();
    let cellpredict = row.insertCell();
    fetch(window.location.href+ "labels/" + sample_n).then(response=>response.text().then(data=>{
      let p = document.createElement('p');
      p.innerHTML = data == 1.0 ? 'real' : 'forge';
      p.className = 'label ' + (data == 1.0 ? 'real' : 'forge');
      celllabel.appendChild(p);
    }))
    fetch(window.location.href+ "predict/" + sample_n).then(response=>response.text().then(data=>{
      let cuttoff = 0.5;
      let p = document.createElement('p');
      let prob = sigmoid(data - 0.7);

      p.innerHTML = Math.round(prob*100)+ '%:' + (prob > cuttoff ? ' real' : ' forge');
      p.className = 'label ' + (prob > cuttoff ? 'real' : 'forge');
      cellpredict.appendChild(p);
    }))


}
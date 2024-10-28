(function(){
    
    const sliders = [...document.querySelectorAll('.desastres__body')];
    const buttonNext = document.querySelector('#next');
    const buttonBefore = document.querySelector('#before');
    let value;   

    buttonNext.addEventListener('click', ()=>{
        changePosition(1);
    });

    buttonBefore.addEventListener('click', ()=>{
        changePosition(-1);
    });

    const changePosition = (add)=>{
        const currentDesastres = document.querySelector('.desastres__body--show').dataset.id;
        value = Number(currentDesastres);
        value+= add;


        sliders[Number(currentDesastres)-1].classList.remove('desastres__body--show');
        if(value === sliders.length+1 || value === 0){
            value = value === 0 ? sliders.length  : 1;
        }

        sliders[value-1].classList.add('desastres__body--show');

    }

})();
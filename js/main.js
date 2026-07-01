(function(){
  var t=document.getElementById("nav-toggle"),n=document.getElementById("nav-links");
  t&&n&&(t.addEventListener("click",function(){
    n.classList.toggle("open"),t.classList.toggle("open")
  }),document.addEventListener("click",function(e){
    n.classList.contains("open")&&!t.contains(e.target)&&!n.contains(e.target)&&(n.classList.remove("open"),t.classList.remove("open"))
  }));

  document.querySelectorAll('.paddle-btn').forEach(function(btn){
    btn.addEventListener('click',function(e){
      e.preventDefault();
      var priceId=this.getAttribute('data-price-id');
      Paddle.Checkout.open({
        items:[{priceId:priceId,quantity:1}]
      });
    });
  });
})();

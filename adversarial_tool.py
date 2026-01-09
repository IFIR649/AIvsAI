import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import sys

class AdversarialInjector:
    def __init__(self, model_name="gpt2"):
        print(f"--- INICIALIZANDO MOTOR: {model_name.upper()} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Hardware detectado: {self.device}")
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(" Sistema listo. Modelo cargado en VRAM/RAM.\n")
            
        except Exception as e:
            print(f"Error crítico cargando modelo: {e}")
            sys.exit(1)

    def _calcular_loss(self, input_ids, target_id):
        """Función interna para calcular la resistencia del modelo"""
        embeddings = self.model.transformer.wte(input_ids).detach()
        embeddings.requires_grad = True
        outputs = self.model(inputs_embeds=embeddings.unsqueeze(0))

        last_token_logits = outputs.logits[0, -1, :]
        loss = F.cross_entropy(last_token_logits.view(1, -1), torch.tensor([target_id]).to(self.device))
        return loss, embeddings

    def atacar(self, prompt, target, iteraciones=50, largo_sufijo=5, verbose=True):
        """
        Ejecuta el algoritmo GCG (Greedy Coordinate Gradient).
        """
        print(f"[*] Iniciando ataque: '{prompt}' -> Objetivo: '{target}'")
        

        target_id = self.tokenizer.encode(target)[0]
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(prompt_ids).to(self.device)
        

        sufijo_ids = torch.randint(33, 126, (largo_sufijo,)).to(self.device)
        
        mejor_loss = float('inf')
        mejor_sufijo_str = ""


        for i in range(iteraciones):

            input_ids = torch.cat([prompt_tensor, sufijo_ids])
            loss, embeddings = self._calcular_loss(input_ids, target_id)
            

            if loss.item() < mejor_loss:
                mejor_loss = loss.item()
                mejor_sufijo_str = self.tokenizer.decode(sufijo_ids)
                if verbose and i % 5 == 0:
                    print(f"Iter {i}: Loss {mejor_loss:.4f} | Sufijo Actual: '{mejor_sufijo_str}'")
                

                if mejor_loss < 0.5:
                    print(f"\n[!] SEGURO ROTO EN ITERACIÓN {i}")
                    break
            
            loss.backward()
            
            grad_sufijo = embeddings.grad[len(prompt_ids):]
            
            with torch.no_grad():
                vocab_weights = self.model.transformer.wte.weight
                scores = -torch.matmul(grad_sufijo, vocab_weights.t())
                
                pos_mutar = random.randint(0, largo_sufijo - 1)
                candidatos = torch.topk(scores[pos_mutar], 30).indices

            loss_min_batch = float('inf')
            mejor_candidato = sufijo_ids[pos_mutar]
            
            for cand in candidatos:
                temp_sufijo = sufijo_ids.clone()
                temp_sufijo[pos_mutar] = cand
                
                with torch.no_grad():
                    temp_inp = torch.cat([prompt_tensor, temp_sufijo])
                    out = self.model(temp_inp.unsqueeze(0))
                    val_loss = F.cross_entropy(out.logits[0, -1, :].view(1, -1), torch.tensor([target_id]).to(self.device))
                
                if val_loss.item() < loss_min_batch:
                    loss_min_batch = val_loss.item()
                    mejor_candidato = cand
            
            sufijo_ids[pos_mutar] = mejor_candidato

        return mejor_sufijo_str, mejor_loss

    def verificar_resultado(self, prompt, sufijo):
        full_input = prompt + sufijo
        inputs = self.tokenizer.encode(full_input, return_tensors="pt").to(self.device)
        out = self.model.generate(inputs, max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

# --- (CLI) ---
def main():
    injector = AdversarialInjector() 
    
    while True:
        print("\n" + "="*40)
        print("  SUITE DE INGENIERÍA ADVERSARIA (GPT-2)")
        print("="*40)
        print("1. Completar Frase (Modo Fácil)")
        print("2. Romper Lógica Matemática (Modo Medio)")
        print("3. Inversión de Sentimiento (Modo Difícil)")
        print("4. Ataque Personalizado (Free Mode)")
        print("5. Salir")
        
        opcion = input("\nSelecciona una opción (1-5): ")
        
        if opcion == "5":
            print("Cerrando sesión...")
            break
            
        prompt_user = ""
        target_user = ""
        iters = 50
        len_suffix = 5
        
        if opcion == "1":
            prompt_user = input("Introduce el inicio (ej: The president is): ")
            target_user = " " + input("Introduce el objetivo (ej: Voldemort): ").strip()
            iters = 30
            len_suffix = 3 
            
        elif opcion == "2":
            prompt_user = input("Introduce la ecuación (ej: 2 + 2 =): ")
            target_user = " " + input("Introduce resultado falso (ej: 5): ").strip()
            iters = 60
            len_suffix = 5 
            
        elif opcion == "3":
            prompt_user = input("Introduce texto negativo: ")
            target_user = " Positive" 
            iters = 100
            len_suffix = 6 
            
        elif opcion == "4":
            prompt_user = input("Prompt: ")
            target_user = " " + input("Target (recuerda el espacio inicial): ").strip()
            iters = int(input("Iteraciones (recomendado 50): "))
            len_suffix = int(input("Longitud del sufijo (recomendado 5): "))
            
        else:
            print("Opción no válida.")
            continue


        print(f"\n---> Configurando ataque: {iters} ciclos, {len_suffix} tokens de ruido.")
        sufijo_ganador, loss_final = injector.atacar(prompt_user, target_user, iteraciones=iters, largo_sufijo=len_suffix)
        
        print("-" * 30)
        print(f"RESULTADO FINAL (Loss: {loss_final:.4f})")
        print(f"Prompt Inyectado: {prompt_user}{sufijo_ganador}")
        print("-" * 30)

        if input("¿Quieres probar la generación real? (s/n): ").lower() == 's':
            resultado = injector.verificar_resultado(prompt_user, sufijo_ganador)
            print(f"\nSalida de la IA:\n{resultado}")

if __name__ == "__main__":
    main()
import flet as ft
import os, sys

async def main(page: ft.Page):
    page.title = 'Test Close - Clique no X'
    page.window.width = 400
    page.window.height = 300
    
    status = ft.Text('Clique no X para testar. Veja _close_event_log.txt', size=14)
    page.add(status)
    
    def on_win_event(e):
        # Log tudo sobre o evento
        info = f'Event class={type(e).__name__}'
        for attr in ['type', 'data', 'name', 'target']:
            if hasattr(e, attr):
                val = getattr(e, attr)
                info += f' | {attr}={repr(val)}'
        print(info, flush=True)
        
        with open('_close_event_log.txt', 'a') as f:
            f.write(info + '\n')
        
        # Checar se é close
        is_close = False
        if hasattr(e, 'type'):
            try:
                is_close = (e.type == ft.WindowEventType.CLOSE)
                print(f'  type check: e.type={repr(e.type)} == CLOSE? {is_close}', flush=True)
            except Exception as ex:
                print(f'  type check error: {ex}', flush=True)
        
        if not is_close and hasattr(e, 'data'):
            is_close = ('close' in str(e.data).lower())
            print(f'  data check: e.data={repr(e.data)} contains close? {is_close}', flush=True)
        
        if is_close:
            print('>>> CLOSE DETECTED! Exiting...', flush=True)
            with open('_close_event_log.txt', 'a') as f:
                f.write('>>> CLOSE DETECTED!\n')
            os._exit(0)
        else:
            print(f'  NOT close, ignoring', flush=True)
    
    page.window.prevent_close = True
    page.window.on_event = on_win_event
    page.update()
    print("Handler registered. Click X to test.", flush=True)

ft.run(main, view=ft.AppView.FLET_APP)

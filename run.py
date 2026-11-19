"""شغّل UniFlow للويب والموبايل على نفس الشبكة."""
import os

import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"UniFlow → http://127.0.0.1:{port}/  (web)")
    print(f"موبايل: افتح من الهاتف http://<IP-جهازك>:{port}/  (نفس الواي فاي)")
    uvicorn.run("main:app", host=host, port=port, reload=False)

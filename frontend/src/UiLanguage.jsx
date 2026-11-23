import React, { createContext, useContext, useEffect, useState } from "react";
import { translate } from "./i18n.js";

const Context = createContext(null);

export function I18nProvider({ children }) {
  const [locale, setLocale] = useState("en");
  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);
  const t = (text, values) => translate(locale, text, values);
  return (
    <Context.Provider value={{ locale, setLocale, t }}>
      {children}
    </Context.Provider>
  );
}

export const useI18n = () => useContext(Context);
